import logging

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    datefmt='%H:%M:%S'
)
LOGGING_LEVEL = logging.DEBUG

DEFAULT_TASKS_FILE = "data/eval_tasks.csv"
DEFAULT_NUM_TASKS = 0  # 0 means all tasks
DEFAULT_MCP_ENABLED = True  # Means we communicate only via MCP
DEFAULT_FHIR_SERVER = "http://fhir-server:8080/fhir"

DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MAX_CONCURRENT = 1
DEFAULT_TASK_TIMEOUT = 120  # seconds
DEFAULT_EVAL_MODEL = "openai/gpt-4o-mini"
