from typing import Any, Optional
from pydantic import BaseModel, HttpUrl, Field, computed_field

class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


class ConversationState(BaseModel):
    """Tracks state during a conversation with the purple agent."""
    iterations: int = 0
    trace: list[dict[str, Any]] = Field(default_factory=list)


class TaskResult(BaseModel):
    """Result from running a single evaluation task."""
    final_answer: Optional[str] = None
    iterations: int = 0
    trace: list[dict[str, Any]] = Field(default_factory=list)
    tools_used: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_fhir_resources: dict[str, list] = Field(default_factory=dict)
    error: Optional[str] = None

    # Evaluation metrics (populated after evaluation)
    correct: Optional[int] = None  # 0 or 1
    precision: Optional[float] = None
    recall: Optional[float] = None

class FHIRAgentBenchResult(BaseModel):
    """Result from medical benchmark evaluation."""
    # Summary metrics
    total_tasks: int
    correct_answers: int
    accuracy: float
    avg_precision: float
    avg_recall: float
    f1_score: float
    time_used: float

    # Per-task details
    task_results: list[TaskResult]
