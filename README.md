# FHIR Agent Evaluator

[![Build](https://github.com/abasit/fhiragentevaluator/actions/workflows/test-and-publish.yml/badge.svg)](https://github.com/abasit/fhiragentevaluator/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

FHIR Agent Evaluator is a benchmark for evaluating medical LLM agents on clinical tasks using FHIR (Fast Healthcare Interoperability Resources) data. It implements the [Agent-to-Agent (A2A) protocol](https://github.com/google/A2A) for standardized agent evaluation.

### Task Categories

The benchmark combines and augments tasks from two established medical agent benchmarks:

- **FHIR-AgentBench** - Retrieval and reasoning tasks
- **MedAgentBench** - Action-oriented clinical tasks
- **Drug Interactions** - Medication conflict detection using FDA label information

### Evaluation Metrics

- **Answer Correctness** - LLM-based semantic comparison with reference answers
- **Action Correctness** - Validation of FHIR POST requests (resource type, parameters)
- **Retrieval Precision/Recall** - Comparison of retrieved FHIR resource IDs against ground truth

## Repository Structure
```
├── src/
│   ├── server.py                    # HTTP server + agent card configuration
│   ├── executor.py                  # A2A request lifecycle and execution
│   ├── agent.py                     # Agent orchestration and decision logic
│   ├── messenger.py                 # A2A messaging abstractions
│   │
│   ├── common/                      # Shared evaluation and benchmarking logic
│   │   ├── eval_metrics.py          # Precision/recall and answer correctness metrics
│   │   ├── evaluation.py            # Evaluation pipeline and result aggregation
│   │   ├── fhir_client.py           # FHIR server HTTP client
│   │   ├── models.py                # Result and task dataclasses
│   │   ├── prompt_builder.py        # Task prompt construction
│   │   └── task_loader.py           # CSV loading, response parsing
│   │
│   └── fhir_mcp/                    # MCP server and agent-callable tools
│       ├── server.py                # MCP server with task-scoped storage
│       ├── utils.py                 # Shared MCP utilities
│       └── tools/
│           ├── __init__.py 
│           ├── fhir_tools.py        # FHIR GET/POST tools
│           ├── medical_codes.py     # Medical code lookup
│           ├── drug_labels.py       # FDA drug label retrieval
│           └── python_executor.py   # Sandboxed Python execution
│
├── Dockerfile                       # Docker configuration
├── pyproject.toml                   # Python dependencies
└── .github/
    └─ workflows/
       └─ test-and-publish.yml       # CI workflow
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ with [uv](https://github.com/astral-sh/uv)
- OpenAI API key (for answer evaluation)

### Quick Start

1. Clone and install dependencies:
```bash
git clone https://github.com/abasit/fhiragentevaluator.git
cd fhiragentevaluator
uv sync
```

2. Configure environment:
```bash
cp sample.env .env
# Edit .env with your OpenAI API key
```

3. Start the FHIR database and green agent:
```bash
docker compose up
```

This starts:
- FHIR server (MIMIC-IV-FHIR) on `http://localhost:8080/fhir`
- Green agent on `http://localhost:9009`


  **Note**: The FHIR database image is ~2GB and may take a few minutes to download and initialize on first run.

4. Verify services are running:
```bash
# Check green agent
curl "http://localhost:9009/.well-known/agent.json"

# Check FHIR server
curl "http://localhost:8080/fhir/Patient?_summary=count"
```

## Running the Benchmark

### Via AgentBeats Platform

Instructions are available at the [AgentBeats platform](https://agentbeats.dev).

The green agent is available at https://agentbeats.dev.

<!-- TODO: Update URL after registration -->

### Locally via A2A Request

Send an evaluation request directly to the green agent:
```bash
curl -X POST http://localhost:9009/ \
  -H "Content-Type: application/json" \
  -d '{
    "participants": {
      "purple_agent": "http://localhost:9010"
    },
    "config": {
      "num_tasks": 5,
      "tasks_file": "data/fhiragentbench_tasks.csv",
      "mcp_enabled": true
    }
  }'
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `num_tasks` | 0 (all) | Number of tasks to run |
| `tasks_file` | `data/fhiragentbench_tasks.csv` | Path to task CSV |
| `mcp_enabled` | true | MCP mode (true) or messaging mode (false) |
| `max_iterations` | 10 | Max agent turns per task |
| `max_concurrent` | 3 | Parallel task execution |

## Citation

If you use this benchmark, please cite:
```bibtex
@software{basit2025fhiragentevaluator,
  title={FHIR Agent Evaluator: An A2A Evaluation Framework for Medical LLM Agents},
  author={Basit, Abdul and Batrakova, Maria},
  url={https://github.com/abasit/fhiragentevaluator},
  year={2025}
}
```
This benchmark builds upon:
```bibtex
@article{jiang2025medagentbench,
  title={MedAgentBench: A Virtual EHR Environment to Benchmark Medical LLM Agents},
  author={Jiang, Yixing and Black, Kameron C and Geng, Gloria and Park, Danny and Zou, James and Ng, Andrew Y and Chen, Jonathan H},
  journal={NEJM AI},
  pages={AIdbp2500144},
  year={2025},
  publisher={Massachusetts Medical Society}
}

@inproceedings{lee2025fhiragentbench,
  title={FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering},
  author={Lee, Gyubok and Bach, Elea and Yang, Eric and Pollard, Tom and JOHNSON, ALISTAIR and Choi, Edward and Lee, Jong Ha and others},
  booktitle={Machine Learning for Health 2025}
}
```

This benchmark uses data from:
- [MIMIC-IV-FHIR](https://physionet.org/content/mimic-iv-fhir/) - PhysioNet Credentialed Health Data License
- [FDA Drug Labels API](https://open.fda.gov/) - Public domain

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
