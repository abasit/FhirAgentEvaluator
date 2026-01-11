import asyncio
import json
import logging
import time
from uuid import uuid4

import pandas as pd
from pydantic import ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from common.models import EvalRequest, TaskResult, ConversationState
from fhir_mcp import get_mcp_server, verify_tool_access
from fhir_mcp.tools import get_tool, SUPPORTED_TYPES


logger = logging.getLogger("fhir_green_agent")

DEFAULT_TASKS_FILE = "data/fhiragentbench_tasks.csv"
DEFAULT_NUM_TASKS = None  # None means all tasks
DEFAULT_MCP_ENABLED = True  # Means we communicate only via MCP
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MAX_CONCURRENT = 3

RESPOND_ACTION_NAME = "response"


class Agent:
    required_roles: list[str] = ["purple_agent"]
    required_config_keys: list[str] = []  # All config is optional

    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Add additional request validation here

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Agent logic

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
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

        # Verify tool access
        logger.info(f"Checking tool access")
        verify_tool_access()
        await updater.update_status(TaskState.working, new_agent_text_message(f"Tool access verified"))
        logger.info(f"Tool access verified")

        # Load tasks
        tasks_df = self._load_tasks(tasks_file, num_tasks)
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

        # Run evaluation

        # Report results
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text="The agent performed well.")),
                Part(root=DataPart(data={
                    # structured assessment results
                }))
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
        """Run all tasks with concurrency control."""
        # Create column for results
        tasks_df["result"] = None

        total_tasks = len(tasks_df)
        semaphore = asyncio.Semaphore(max_concurrent)

        # Track progress
        completed = 0
        succeeded = 0
        failed = 0
        start_time = time.time()

        async def run_with_semaphore(idx: int, task) -> tuple[int, TaskResult]:
            async with semaphore:
                task_start = time.time()
                logger.info(f"[Task {idx}] Starting")

                result = await self._run_single_task(
                    purple_agent_url=purple_agent_url,
                    task_idx=idx,
                    question=task.question_with_context,
                    max_iterations=max_iterations,
                    mcp_enabled=mcp_enabled,
                )

                elapsed = time.time() - task_start
                if result and result.error:
                    logger.warning(f"[Task {idx}] Failed in {elapsed:.1f}s: {result.error}")
                else:
                    logger.info(f"[Task {idx}] Completed in {elapsed:.1f}s")

                return idx, result

        # Create all task coroutines
        coroutines = [
            run_with_semaphore(i, task)
            for i, task in enumerate(tasks_df.itertuples(index=True))
        ]

        # Run and process as each completes
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
                    f"ETA: {eta:.0f}s"
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
        """Run a single task against the purple agent."""
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
        """Run a single task in MCP mode - single round trip."""
        state = ConversationState()

        mcp_task_id = str(uuid4())
        system_prompt = self._build_task_prompt_mcp(mcp_task_id)

        state.trace.append({"role": "system", "content": system_prompt})
        state.trace.append({"role": "user", "content": question})

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
            return self._make_result(state, mcp_task_id, error=f"Communication error\n{str(e)}")

        # Parse response
        try:
            parsed_response = self._parse_agent_response(response_text)
            logger.debug(f"[Task {task_idx}] Parsed response: {parsed_response}")
        except json.JSONDecodeError as e:
            logger.error(f"[Task {task_idx}] Parse error: {e}")
            return self._make_result(state, mcp_task_id, error=f"Failed to parse response\n{str(e)}")

        action = parsed_response[0] if parsed_response else {}
        action_name = action.get("name", "").lower()
        action_kwargs = action.get("kwargs", {})

        if action_name == RESPOND_ACTION_NAME:
            content = action_kwargs.get("content", "")

            if self._is_final_answer(content):
                logger.info(f"[Task {task_idx}] Got final answer after {state.iterations} iterations")
                return self._make_result(state, mcp_task_id, final_answer=content)
            else:
                logger.warning(f"[Task {task_idx}] Response without final answer")
                return self._make_result(state, mcp_task_id, error=f"Response without final answer\n{content}")

        logger.warning(f"[Task {task_idx}] Unknown action: {action_name}")
        return self._make_result(state, mcp_task_id, error=f"Unknown action: {action_name}")

    async def _run_single_task_messaging(
            self,
            purple_agent_url: str,
            task_idx: int,
            question: str,
            max_iterations: int,
    ) -> TaskResult:
        """Run a single task in messaging mode - iterative tool calls."""
        from fhir_mcp.server import current_task_id

        state = ConversationState()
        mcp_server = get_mcp_server()
        mcp_task_id = str(uuid4())

        system_prompt = await self._build_task_prompt_messaging()

        state.trace.append({"role": "system", "content": system_prompt})
        state.trace.append({"role": "user", "content": question})

        message_content = f"{system_prompt}\n\n{question}"
        is_first_message = True

        # Set task context for the entire task
        token = current_task_id.set(mcp_task_id)

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
                    return self._make_result(state, mcp_task_id, error=f"Communication error\n{str(e)}")

                # Parse response
                try:
                    parsed_response = self._parse_agent_response(response_text)
                    logger.debug(f"[Task {task_idx}] Parsed response: {parsed_response}")
                except json.JSONDecodeError as e:
                    logger.error(f"[Task {task_idx}] Parse error: {e}")
                    return self._make_result(state, mcp_task_id, error=f"Failed to parse response\n{str(e)}")

                action = parsed_response[0] if parsed_response else {}
                action_name = action.get("name", "").lower()
                action_kwargs = action.get("kwargs", {})

                if action_name == RESPOND_ACTION_NAME:
                    content = action_kwargs.get("content", "")

                    if self._is_final_answer(content):
                        logger.info(f"[Task {task_idx}] Got final answer after {state.iterations} iterations")
                        return self._make_result(state, mcp_task_id, final_answer=content)
                    else:
                        logger.warning(f"[Task {task_idx}] Response without final answer")
                        return self._make_result(state, mcp_task_id, error=f"Response without final answer\n{content}")
                else:
                    tool_name, tool_args = action_name, action_kwargs
                    logger.info(f"[Task {task_idx}] Calling tool: {tool_name} with args: {tool_args}")

                    try:
                        tool_output = self._execute_tool(tool_name, tool_args)
                        tool_output_str = str(tool_output)
                        logger.debug(f"[Task {task_idx}] Tool returned:\n{tool_output_str}")
                        message_content = tool_output_str
                    except Exception as e:
                        logger.error(f"[Task {task_idx}] Tool {tool_name} failed: {e}")
                        return self._make_result(state, mcp_task_id, error=f"Tool execution failed\n{str(e)}")

            logger.warning(f"[Task {task_idx}] Max iterations ({max_iterations}) reached")
            return self._make_result(state, mcp_task_id, error="Max iterations reached")

        finally:
            current_task_id.reset(token)

    @staticmethod
    def _build_task_prompt_mcp(mcp_task_id: str) -> str:
        """
        Build the system prompt for the purple agent.
        """
        mcp_server = get_mcp_server()
        mcp_url = f"{mcp_server.base_url}/tasks/{mcp_task_id}/mcp"

        return f"""
Use tools available on the MCP server available at: {mcp_url}

Task instructions:
    - Always use the Patient FHIR ID provided in context; ignore numeric patient IDs in the question text.
    - Retrieve only what you need; avoid broad/all-observation pulls (no bulk fetches).
    - Provide all answers found in the same format as the retrieved data. If you cannot find the answer, clearly say so.
    - Do not guess attributes and do not repeat the same failing action.

    Response format:
    - Wrap your response in <json>...</json>.
    - For final answer use:
    {{"name": "{RESPOND_ACTION_NAME}", "kwargs": {{"content": "The final answer is: ..."}}}}

    IMPORTANT: The content for your final message must start with 'The final answer is:' followed by your conclusion.
"""

    @staticmethod
    async def _build_task_prompt_messaging() -> str:
        """Build the system prompt with tool descriptions."""
        mcp_server = get_mcp_server()
        all_tools = await mcp_server.get_tool_definitions()

        return f"""
Here's a list of tools you can use (you can use one tool at a time):
{all_tools}

Available FHIR resource types: {', '.join(SUPPORTED_TYPES)}.
You can only call on these FHIR resources types for retrieval.

To answer questions about patient data:
1. Always use the Patient FHIR ID provided in context; do not use the numeric patient ID from the question text.
2. Use lookup_medical_code to find codes for medical items, labs, and procedures
3. Use fhir_request_get to get data from the FHIR server
4. Analyze the retrieved data to answer the question

If there are multiple answers, provide all of them.
When you provide answers, make sure to provide them in the same format as they are in the retrieved data. If multiple answers are provided, provide them all in a list.
If you cannot find the answer or relevant patient data, clearly state that you cannot find the information.
Do not guess attributes; instead, use the provided tool to retrieve the data.
Do not get stuck or repeat the same action.

Please respond in the JSON format. Please wrap the JSON part with <json>...</json> tags.
The JSON can be either a single action or a list of actions.

Tool Call:
{{"name": "tool_name", "kwargs": {{"arg": "value"}}}}

For final response:
{{"name": "{RESPOND_ACTION_NAME}", "kwargs": {{"content": "The final answer is: ..."}}}}

IMPORTANT: The content for your final message must start with 'The final answer is:' followed by your conclusion. This is required for proper processing.
"""

    @staticmethod
    def _make_result(
            state: ConversationState,
            mcp_task_id: str,
            final_answer: str = None,
            error: str = None,
    ) -> TaskResult:
        mcp_server = get_mcp_server()

        result = TaskResult(
            final_answer=final_answer,
            tools_used=mcp_server.get_tool_logs(mcp_task_id),
            retrieved_fhir_resources=mcp_server.get_task_resources(mcp_task_id),
            trace=state.trace,
            iterations=state.iterations,
            error=error,
        )

        mcp_server.clear_task(mcp_task_id)

        return result

    @staticmethod
    def _load_tasks(tasks_file: str, num_tasks: int = None) -> pd.DataFrame:
        """Load tasks from file."""
        tasks_df = pd.read_csv(tasks_file)[
            ["question_id", "question", "true_answer", "assumption", "patient_fhir_id", "true_fhir_ids"]
        ]

        if num_tasks is not None:
            tasks_df = tasks_df[:num_tasks].copy()

        # Add question_with_context
        def create_input_str(row):
            input_str = f"Question: {row['question']}\nContext:"
            input_str += f"\nPatient FHIR ID is {row['patient_fhir_id']}."
            if pd.notnull(row['assumption']):
                input_str += f"\n{row['assumption']}"
            return input_str

        tasks_df["question_with_context"] = tasks_df.apply(create_input_str, axis=1)

        return tasks_df

    @staticmethod
    def _parse_agent_response(response_text: str) -> list:
        import re

        json_str = None
        match = re.search(r'<json>\s*(.*?)\s*</json>', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if match:
                    json_str = match.group(1)

        if json_str:
            parsed = json.loads(json_str)
        else:
            parsed = json.loads(response_text)

        if isinstance(parsed, list):
            return parsed
        else:
            return [parsed]

    @staticmethod
    def _is_final_answer(content: str) -> bool:
        if not content:
            return False
        return "the final answer is:" in content.lower().strip()

    @staticmethod
    def _execute_tool(tool_name: str, tool_args: dict) -> dict:
        tool_function = get_tool(tool_name)
        result = tool_function(**tool_args)

        # Log the tool call (uses current_task_id context)
        get_mcp_server().log_tool_call(tool_name, tool_args, result)

        return result