import argparse
import logging
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor
from fhir_mcp import init_mcp_server


# Configure logging
logging.basicConfig(level=logging.WARNING)

logging.getLogger("fhir_green_agent").setLevel(logging.INFO)
logging.getLogger("fhir_mcp").setLevel(logging.INFO)
logging.getLogger("fhir_common").setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    base_url = args.card_url or f"http://{args.host}:{args.port}"

    # Agent card config
    skill = AgentSkill(
        id="fhir_agent_evaluator",
        name="Evaluates FHIR agents",
        description="Evaluates medical agents interacting with FHIR databases for clinical tasks",
        tags=["clinical tasks", "medical agents", "fhir agent"],
        examples=["""
{
  "participants": {
    "purple_agent": "http://localhost:9009"
  },
  "config": {
    "num_tasks": 3,                                  # Optional, defaults to None (all tasks)
    "tasks_file": "data/fhiragentbench_tasks.csv",   # Optional, defaults to "data/fhiragentbench_tasks.csv"
    "mcp_enabled": true,                             # Optional, defaults to true
    "max_iterations": 10,                            # Optional, defaults to 10
    "max_concurrent": 3,                             # Optional, defaults to 3
  }
}
"""]
    )

    agent_card = AgentCard(
        name="FHIR Agent Evaluator",
        description="Evaluates medical agents interacting with FHIR databases for clinical tasks",
        url=base_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    # Create MCP server
    mcp_server = init_mcp_server(base_url=base_url)

    # Create A2A server
    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).build()

    root_app = Starlette(
        routes=[
            mcp_server.get_routes(),
            Mount("/", app=a2a_app),
        ],
        lifespan=mcp_server.lifespan,
    )

    uvicorn.run(root_app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()