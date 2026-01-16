"""
FHIR Agent Benchmark - Green Agent

Orchestrates evaluation of purple (participant) agents on FHIR-based medical tasks.
Supports two modes:
- MCP mode: Single round-trip, agent connects to MCP server for tools
- Messaging mode: Iterative tool calls via A2A messaging

The agent loads tasks, runs them against the purple agent, and evaluates results
using retrieval metrics (precision/recall) and answer correctness.
"""
import asyncio
import json
import logging
import time
import warnings

import pandas as pd
from pydantic import ValidationError
from uuid import uuid4

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from common.models import EvalRequest, TaskResult, ConversationState
from common.evaluation import evaluate_results
from common.prompt_builder import build_task_prompt_messaging, RESPOND_ACTION_NAME, build_task_prompt_mcp
from common.task_loader import load_tasks, make_result, is_final_answer, parse_agent_response
from fhir_mcp import verify_tool_access, execute_tool
from fhir_mcp.server import current_task_id

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
logger = logging.getLogger("fhir_green_agent")

DEFAULT_TASKS_FILE = "data/all_tasks.csv"
DEFAULT_NUM_TASKS = 0  # 0 means all tasks
DEFAULT_MCP_ENABLED = True  # Means we communicate only via MCP
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MAX_CONCURRENT = 3
DEFAULT_EVAL_MODEL = "openai/gpt-4o-mini"
DEFAULT_TASK_TIMEOUT = 60  # seconds

def format_eta(seconds: float) -> str:
    """
    Format an ETA in seconds into a human-readable string.
    """
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"

    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"

class Agent:
    """
    Green agent that evaluates purple agents on FHIR benchmark tasks.

    Handles request validation, task execution (concurrent), and result evaluation.
    """
    required_roles: list[str] = ["purple_agent"]
    required_config_keys: list[str] = []  # All config is optional

    def __init__(self):
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate that request has required participants and config keys."""
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Main evaluation entry point.

        Args:
            message: EvalRequest JSON with participants and config
            updater: Task updater for progress and results
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        logger.info(f"Received request: {request}")

        # Get config with defaults
        num_tasks = request.config.get("num_tasks", DEFAULT_NUM_TASKS)
        tasks_file = request.config.get("tasks_file", DEFAULT_TASKS_FILE)
        mcp_enabled = request.config.get("mcp_enabled", DEFAULT_MCP_ENABLED)
        max_iterations = request.config.get("max_iterations", DEFAULT_MAX_ITERATIONS)
        max_concurrent = request.config.get("max_concurrent", DEFAULT_MAX_CONCURRENT)
        eval_model = DEFAULT_EVAL_MODEL

        # Verify tool access
        logger.info("Verifying FHIR database connection...")
        verify_tool_access()
        await updater.update_status(TaskState.working, new_agent_text_message("FHIR database connected"))
        logger.info("FHIR database connected")

        # Load tasks
        tasks_df = load_tasks(tasks_file, num_tasks)
        total_tasks = len(tasks_df)

        await updater.update_status(TaskState.working, new_agent_text_message(f"Loaded {total_tasks} tasks"))
        logger.info(f"Loaded {total_tasks} tasks")

        # Get purple agent URL
        purple_agent_url = str(request.participants["purple_agent"])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of purple agent at {purple_agent_url}")
        )
        logger.info(f"Starting evaluation of purple agent at {purple_agent_url}")

        start_time = time.time()

        # Run all tasks
        results_df = await self._run_all_tasks(
            tasks_df=tasks_df,
            purple_agent_url=purple_agent_url,
            max_iterations=max_iterations,
            mcp_enabled=mcp_enabled,
            max_concurrent=max_concurrent,
            updater=updater,
        )

        time_used = time.time() - start_time
        logger.info(f"Task execution completed in {time_used:.1f}s")

        # Run evaluation
        await updater.update_status(TaskState.working, new_agent_text_message("Evaluating results..."))

        eval_result = await evaluate_results(
            tasks_df=results_df,
            eval_model=eval_model,
            max_concurrent=max_concurrent,
            time_used=time_used,
        )

        # Report results
        summary = (
            f"Evaluation complete:\n"
            f"- Total tasks: {eval_result.total_tasks}\n"
            f"- Correct answers: {eval_result.correct_answers} ({eval_result.accuracy * 100:.1f}%)\n"
            f"- Precision: {eval_result.avg_precision:.4f}\n"
            f"- Recall: {eval_result.avg_recall:.4f}\n"
            f"- F1 Score: {eval_result.f1_score:.4f}\n"
            f"- Time: {eval_result.time_used:.1f}s"
        )

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary)),
                Part(root=DataPart(data=eval_result.model_dump())),
            ],
            name="Result",
        )

    async def _run_all_tasks(
            self,
            tasks_df: pd.DataFrame,
            purple_agent_url: str,
            max_iterations: int,
            mcp_enabled: bool,
            max_concurrent: int,
            updater: TaskUpdater,
    ) -> pd.DataFrame:
        """Run all tasks concurrently with progress updates."""
        tasks_df["result"] = None

        total_tasks = len(tasks_df)
        semaphore = asyncio.Semaphore(max_concurrent)

        completed = 0
        succeeded = 0
        failed = 0
        start_time = time.time()

        async def run_with_semaphore(idx: int, task) -> tuple[int, TaskResult]:
            async with semaphore:
                task_start = time.time()
                logger.info(f"[Task {idx}] Starting")

                try:
                    result = await asyncio.wait_for(
                        self._run_single_task(
                            purple_agent_url=purple_agent_url,
                            task_idx=idx,
                            question=task.question_with_context,
                            max_iterations=max_iterations,
                            mcp_enabled=mcp_enabled,
                        ),
                        timeout=DEFAULT_TASK_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"[Task {idx}] Timed out after {DEFAULT_TASK_TIMEOUT}s")
                    result = TaskResult(error=f"Task timed out after {DEFAULT_TASK_TIMEOUT}s")

                result.question = task.question_with_context
                result.question_id = task.question_id

                elapsed = time.time() - task_start
                if result and result.error:
                    logger.warning(f"[Task {idx}] Failed in {elapsed:.1f}s: {result.error}")
                else:
                    logger.info(f"[Task {idx}] Completed in {elapsed:.1f}s")

                return idx, result
        coroutines = [
            run_with_semaphore(i, task)
            for i, task in enumerate(tasks_df.itertuples(index=True))
        ]

        for coro in asyncio.as_completed(coroutines):
            try:
                idx, result = await coro
                completed += 1

                if result.error:
                    failed += 1
                else:
                    succeeded += 1

                # Progress update
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_tasks - completed) / rate if rate > 0 else 0

                progress_msg = (
                    f"Progress: {completed}/{total_tasks} "
                    f"({succeeded} ok, {failed} failed) "
                    f"ETA: {format_eta(eta)}"
                )
                await updater.update_status(TaskState.working, new_agent_text_message(progress_msg))

                # Store result
                tasks_df.at[tasks_df.index[idx], "result"] = result

            except Exception as e:
                completed += 1
                failed += 1
                logger.exception(f"Task execution failed: {e}")

        return tasks_df

    async def _run_single_task(
            self,
            purple_agent_url: str,
            task_idx: int,
            question: str,
            max_iterations: int,
            mcp_enabled: bool,
    ) -> TaskResult:
        """Route task to MCP or messaging mode."""
        if mcp_enabled:
            return await self._run_single_task_mcp(
                purple_agent_url=purple_agent_url,
                task_idx=task_idx,
                question=question,
            )
        else:
            return await self._run_single_task_messaging(
                purple_agent_url=purple_agent_url,
                task_idx=task_idx,
                question=question,
                max_iterations=max_iterations,
            )

    async def _run_single_task_mcp(
            self,
            purple_agent_url: str,
            task_idx: int,
            question: str,
    ) -> TaskResult:
        """Run task in MCP mode - agent connects to MCP server for tools."""
        state = ConversationState()

        mcp_task_id = str(uuid4())
        system_prompt = build_task_prompt_mcp(mcp_task_id)

        state.trace.append({"role": "task prompt", "content": question})

        message_content = f"{system_prompt}\n\n{question}"

        # Send message and receive response
        try:
            state.iterations += 1
            logger.debug(f"[Task {task_idx}] Sending:\n{message_content}")

            response_text = await self.messenger.talk_to_agent(
                message=message_content,
                url=purple_agent_url,
                task_id=task_idx,
                new_conversation=True,
            )

            state.trace.append({"role": "agent", "content": response_text})
            logger.debug(f"[Task {task_idx}] Received:\n{response_text}")
        except Exception as e:
            logger.error(f"[Task {task_idx}] Communication error: {e}")
            return make_result(state, mcp_task_id, error=f"Error communicating with purple agent")

        # Parse response
        try:
            parsed_response = parse_agent_response(response_text)
            logger.debug(f"[Task {task_idx}] Parsed response: {parsed_response}")
        except json.JSONDecodeError as e:
            logger.error(f"[Task {task_idx}] Parse error: {e}")
            return make_result(state, mcp_task_id, error=f"Failed to parse purple agent response:\n{response_text}")

        action = parsed_response[0] if parsed_response else {}
        action_name = action.get("name", "").lower()
        action_kwargs = action.get("kwargs", {})

        if action_name == RESPOND_ACTION_NAME:
            content = action_kwargs.get("content", "")

            if is_final_answer(content):
                logger.info(f"[Task {task_idx}] Got final answer after {state.iterations} iterations")
                return make_result(state, mcp_task_id, final_answer=content)
            else:
                logger.warning(f"[Task {task_idx}] Response without final answer")
                return make_result(state, mcp_task_id, error=f"Response without final answer\n{content}")

        logger.warning(f"[Task {task_idx}] Unknown action: {action_name}")
        return make_result(state, mcp_task_id, error=f"Unknown action: {action_name}")

    async def _run_single_task_messaging(
            self,
            purple_agent_url: str,
            task_idx: int,
            question: str,
            max_iterations: int,
    ) -> TaskResult:
        """Run task in messaging mode - green agent executes tools iteratively."""
        # Set task context for the entire task
        mcp_task_id = str(uuid4())
        token = current_task_id.set(mcp_task_id)

        state = ConversationState()

        system_prompt = await build_task_prompt_messaging()

        state.trace.append({"role": "task prompt", "content": question})

        message_content = f"{system_prompt}\n\n{question}"
        is_first_message = True

        try:
            while state.iterations < max_iterations:
                state.iterations += 1
                logger.info(f"[Task {task_idx}] Iteration {state.iterations}/{max_iterations}")

                # Send message and receive response
                try:
                    logger.debug(f"[Task {task_idx}] Sending:\n{message_content}")

                    response_text = await self.messenger.talk_to_agent(
                        message=message_content,
                        url=purple_agent_url,
                        task_id=task_idx,
                        new_conversation=is_first_message,
                    )
                    is_first_message = False

                    state.trace.append({"role": "agent", "content": response_text})
                    logger.debug(f"[Task {task_idx}] Received:\n{response_text}")
                except Exception as e:
                    logger.error(f"[Task {task_idx}] Communication error: {e}")
                    return make_result(state, mcp_task_id, error=f"Error communicating with purple agent")

                # Parse response
                try:
                    parsed_response = parse_agent_response(response_text)
                    logger.debug(f"[Task {task_idx}] Parsed response: {parsed_response}")
                except json.JSONDecodeError as e:
                    logger.error(f"[Task {task_idx}] Parse error: {e}")
                    return make_result(state, mcp_task_id, error=f"Failed to parse purple agent response:\n{response_text}")

                action = parsed_response[0] if parsed_response else {}
                action_name = action.get("name", "").lower()
                action_kwargs = action.get("kwargs", {})

                if action_name == RESPOND_ACTION_NAME:
                    content = action_kwargs.get("content", "")

                    if is_final_answer(content):
                        logger.info(f"[Task {task_idx}] Got final answer after {state.iterations} iterations")
                        return make_result(state, mcp_task_id, final_answer=content)
                    else:
                        logger.warning(f"[Task {task_idx}] Response without final answer")
                        return make_result(state, mcp_task_id, error=f"Response without final answer\n{content}")
                else:
                    tool_name, tool_args = action_name, action_kwargs
                    logger.info(f"[Task {task_idx}] Calling tool: {tool_name} with args: {tool_args}")

                    try:
                        tool_output = execute_tool(tool_name, tool_args)
                        tool_output_str = str(tool_output)
                        logger.debug(f"[Task {task_idx}] Tool returned:\n{tool_output_str}")
                        message_content = tool_output_str
                        state.trace.append({"role": "tool call result", "content": message_content})
                    except Exception as e:
                        logger.error(f"[Task {task_idx}] Tool {tool_name} failed: {e}")
                        return make_result(state, mcp_task_id, error=f"Tool execution failed")

            logger.warning(f"[Task {task_idx}] Max iterations ({max_iterations}) reached")
            return make_result(state, mcp_task_id, error="Max iterations reached")

        finally:
            current_task_id.reset(token)
