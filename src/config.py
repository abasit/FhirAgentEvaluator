import logging

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    datefmt='%H:%M:%S'
)
LOGGING_LEVEL = logging.INFO

DEFAULT_TASKS_FILE = "data/eval_tasks.csv"
DEFAULT_NUM_TASKS = 0  # 0 means all tasks
DEFAULT_TASK_TIMEOUT = 120  # seconds
DEFAULT_MCP_ENABLED = True  # Means tools are accessed via MCP
DEFAULT_MAX_ITERATIONS = 10 # only valid in messaging mode (when MCP enabled is false)
DEFAULT_FHIR_SERVER = "http://fhir-server:8080/fhir"


# Not to be changed
DEFAULT_EVAL_MODEL = "openai/gpt-4o-mini"
DEFAULT_MAX_CONCURRENT = 1 # Concurrency support disabled. MCP library is unable to handle concurrent requests, causing task failures.
