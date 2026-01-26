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
import os
import time
import warnings

from pydantic import ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from common.models import EvalRequest, TaskResult, ConversationState, Task
from common.evaluation import evaluate_results
from common.prompt_builder import build_task_prompt_messaging, RESPOND_ACTION_NAME, build_task_prompt_mcp
from common.utils import load_tasks, make_result, parse_agent_response, format_time
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
DEFAULT_TASK_TIMEOUT = 120  # seconds


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
        tasks = load_tasks(tasks_file, num_tasks)
        total_tasks = len(tasks)

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
        tasks = await self._run_all_tasks(
            tasks=tasks,
            purple_agent_url=purple_agent_url,
            max_iterations=max_iterations,
            mcp_enabled=mcp_enabled,
            max_concurrent=max_concurrent,
            updater=updater,
        )

        time_used = time.time() - start_time
        logger.info(f"Task execution completed in {format_time(time_used)}")

        # Run evaluation
        await updater.update_status(TaskState.working, new_agent_text_message("Evaluating results..."))

        eval_result = await evaluate_results(
            tasks=tasks,
            eval_model=eval_model,
            max_concurrent=max_concurrent,
            time_used=time_used,
        )

        # Report results
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=eval_result.summary())),
                Part(root=DataPart(data=eval_result.model_dump())),
            ],
            name="Result",
        )

        if os.environ.get("DEBUG_TRACES"):
            debug_output = [
                {
                    "question_id": task.question_id,
                    "question": task.question_with_context,
                    "final_answer": task.result.final_answer if task.result else None,
                    "true_answer": task.true_answer,
                    "correct": task.result.correct if task.result else None,
                    "trace": task.result.trace if task.result else [],
                    "tools_used": task.result.tools_used if task.result else [],
                }
                for task in tasks
            ]
            with open("debug_traces.json", "w") as f:
                json.dump(debug_output, f, indent=2)
            logger.info(f"Wrote debug traces to debug_traces.json")

    async def _run_all_tasks(
            self,
            tasks: list[Task],
            purple_agent_url: str,
            max_iterations: int,
            mcp_enabled: bool,
            max_concurrent: int,
            updater: TaskUpdater,
    ) -> list[Task]:
        """Run all tasks concurrently with progress updates."""
        total_tasks = len(tasks)
        semaphore = asyncio.Semaphore(max_concurrent)

        completed = 0
        succeeded = 0
        failed = 0
        start_time = time.time()

        async def run_with_semaphore(task: Task) -> None:
            nonlocal completed, succeeded, failed

            async with semaphore:
                task_start = time.time()
                logger.debug(f"[Task {task.question_id}] Starting")

                try:
                    result = await asyncio.wait_for(
                        self._run_single_task(
                            purple_agent_url=purple_agent_url,
                            task=task,
                            max_iterations=max_iterations,
                            mcp_enabled=mcp_enabled,
                        ),
                        timeout=DEFAULT_TASK_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"[Task {task.question_id}] Timed out after {DEFAULT_TASK_TIMEOUT}s")
                    result = TaskResult(error=f"Task timed out after {DEFAULT_TASK_TIMEOUT}s")
                except Exception as e:
                    logger.exception(f"[Task {task.question_id}] Unexpected error: {e}")
                    result = TaskResult(error=f"Unexpected error: {e}")

                task.result = result

                elapsed = time.time() - task_start
                if result.error:
                    failed += 1
                    logger.warning(f"[Task {task.question_id}] Failed in {elapsed:.1f}s: {result.error}")
                else:
                    succeeded += 1
                    logger.debug(f"[Task {task.question_id}] Completed in {elapsed:.1f}s")

                completed += 1
                if completed % 50 == 0 or completed == total_tasks:
                    elapsed_total = time.time() - start_time
                    rate = completed / elapsed_total if elapsed_total > 0 else 0
                    eta = (total_tasks - completed) / rate if rate > 0 else 0

                    progress_msg = (
                        f"Progress: {completed}/{total_tasks} "
                        f"({succeeded} ok, {failed} failed) "
                        f"ETA: {format_time(eta)}"
                    )
                    logger.info(progress_msg)
                    await updater.update_status(TaskState.working, new_agent_text_message(progress_msg))

        await asyncio.gather(*[run_with_semaphore(task) for task in tasks])

        return tasks

    async def _run_single_task(
            self,
            purple_agent_url: str,
            task: Task,
            max_iterations: int,
            mcp_enabled: bool,
    ) -> TaskResult:
        """Route task to MCP or messaging mode."""
        if mcp_enabled:
            return await self._run_single_task_mcp(
                purple_agent_url=purple_agent_url,
                task=task,
            )
        else:
            return await self._run_single_task_messaging(
                purple_agent_url=purple_agent_url,
                task=task,
                max_iterations=max_iterations,
            )

    async def _run_single_task_mcp(
            self,
            purple_agent_url: str,
            task: Task,
    ) -> TaskResult:
        """Run task in MCP mode - agent connects to MCP server for tools."""
        state = ConversationState()

        q_id = task.question_id
        mcp_task_id = task.question_id
        system_prompt = build_task_prompt_mcp(mcp_task_id)

        state.trace.append({"role": "task prompt", "content": task.question_with_context})

        message_content = f"{system_prompt}\n\n{task.question_with_context}"

        # Send message and receive response
        try:
            state.iterations += 1
            # logger.debug(f"[Task {q_id}] Sending:\n{message_content}")

            response_text = await self.messenger.talk_to_agent(
                message=message_content,
                url=purple_agent_url,
                task_id=q_id,
                new_conversation=True,
            )

            state.trace.append({"role": "agent", "content": response_text})
            logger.debug(f"[Task {q_id}] Received:\n{response_text}")
        except Exception as e:
            logger.error(f"[Task {q_id}] Communication error: {e}")
            return make_result(state, mcp_task_id, error=f"Error communicating with purple agent")

        # Parse response
        try:
            parsed_response = parse_agent_response(response_text)
            logger.debug(f"[Task {q_id}] Parsed response: {parsed_response}")
        except json.JSONDecodeError as e:
            logger.error(f"[Task {q_id}] Parse error: {e}")
            return make_result(state, mcp_task_id, error=f"Failed to parse purple agent response:\n{response_text}")

        action = parsed_response[0] if parsed_response else {}
        action_name = action.get("name", "").lower()
        action_kwargs = action.get("kwargs", {})

        if action_name == RESPOND_ACTION_NAME:
            content = action_kwargs.get("content", "")
            return make_result(state, mcp_task_id, final_answer=content)

        logger.warning(f"[Task {q_id}] Unexpected action \"{action_name}\" for MCP mode")
        return make_result(state, mcp_task_id, error=f"[Task {q_id}] Unexpected action \"{action_name}\" for MCP mode")

    async def _run_single_task_messaging(
            self,
            purple_agent_url: str,
            task : Task,
            max_iterations: int,
    ) -> TaskResult:
        """Run task in messaging mode - green agent executes tools iteratively."""
        # Set task context for the entire task
        q_id = task.question_id
        mcp_task_id = task.question_id
        token = current_task_id.set(mcp_task_id)

        state = ConversationState()

        system_prompt = await build_task_prompt_messaging()

        state.trace.append({"role": "task prompt", "content": task.question_with_context})

        message_content = f"{system_prompt}\n\n{task.question_with_context}"
        is_first_message = True

        try:
            while state.iterations < max_iterations:
                state.iterations += 1
                logger.debug(f"[Task {q_id}] Iteration {state.iterations}/{max_iterations}")

                # Send message and receive response
                try:
                    # logger.debug(f"[Task {q_id}] Sending:\n{message_content}")

                    response_text = await self.messenger.talk_to_agent(
                        message=message_content,
                        url=purple_agent_url,
                        task_id=q_id,
                        new_conversation=is_first_message,
                    )
                    is_first_message = False

                    state.trace.append({"role": "agent", "content": response_text})
                    logger.debug(f"[Task {q_id}] Received:\n{response_text}")
                except Exception as e:
                    logger.error(f"[Task {q_id}] Communication error: {e}")
                    return make_result(state, mcp_task_id, error=f"Error communicating with purple agent")

                # Parse response
                try:
                    parsed_response = parse_agent_response(response_text)
                    logger.debug(f"[Task {q_id}] Parsed response: {parsed_response}")
                except json.JSONDecodeError as e:
                    logger.error(f"[Task {q_id}] Parse error: {e}")
                    return make_result(state, mcp_task_id, error=f"Failed to parse purple agent response:\n{response_text}")

                action = parsed_response[0] if parsed_response else {}
                action_name = action.get("name", "").lower()
                action_kwargs = action.get("kwargs", {})

                if action_name == RESPOND_ACTION_NAME:
                    content = action_kwargs.get("content", "")
                    return make_result(state, mcp_task_id, final_answer=content)
                else:
                    tool_name, tool_args = action_name, action_kwargs
                    logger.debug(f"[Task {q_id}] Calling tool: {tool_name} with args: {tool_args}")

                    try:
                        tool_output = execute_tool(tool_name, tool_args)
                        tool_output_str = str(tool_output)
                        logger.debug(f"[Task {q_id}] Tool returned:\n{tool_output_str}")
                        message_content = tool_output_str
                        state.trace.append({"role": "tool call result", "content": message_content})
                    except Exception as e:
                        logger.error(f"[Task {q_id}] Tool {tool_name} failed: {e}")
                        return make_result(state, mcp_task_id, error=f"Tool execution failed")

            logger.warning(f"[Task {q_id}] Max iterations ({max_iterations}) reached")
            return make_result(state, mcp_task_id, error="Max iterations reached")

        finally:
            current_task_id.reset(token)
