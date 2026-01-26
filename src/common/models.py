"""
Data models for FHIR Agent Benchmark.

Defines request/response schemas and result structures used throughout evaluation.
"""

from typing import Any, Optional
from pydantic import BaseModel, HttpUrl, Field


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class ConversationState(BaseModel):
    """Tracks state during a conversation with the purple agent."""
    iterations: int = 0
    trace: list[dict[str, Any]] = Field(default_factory=list)


class TaskResult(BaseModel):
    """Result from running a single evaluation task."""
    final_answer: Optional[str] = None
    error: Optional[str] = None

    # Debug info
    iterations: int = 0
    trace: list[dict[str, Any]] = Field(default_factory=list)
    tools_used: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_fhir_ids: dict[str, list] = Field(default_factory=dict)

    # Populated after evaluation
    correct: Optional[int] = None
    precision: Optional[float] = None
    recall: Optional[float] = None


class Task(BaseModel):
    """A single evaluation task."""
    question_id: str
    question_with_context: str
    true_answer: str
    true_fhir_ids: dict[str, list] = Field(default_factory=dict)
    task_type: str
    expected_actions: list[dict] = Field(default_factory=list)
    result: Optional[TaskResult] = None


class TaskOutput(BaseModel):
    """Flattened task + result for final output."""
    question_id: str
    question: str
    true_answer: str
    final_answer: Optional[str] = None
    correct: Optional[int] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    error: Optional[str] = None


class FHIRAgentBenchResult(BaseModel):
    """Aggregated benchmark results with per-task details."""
    total_tasks: int
    correct_answers: int
    accuracy: float
    avg_precision: float
    avg_recall: float
    f1_score: float
    time_used: str
    task_results: list[TaskOutput]

    def summary(self) -> str:
        from common.utils import format_time  # lazy import
        return (
            f"Evaluation complete:\n"
            f"- Total tasks: {self.total_tasks}\n"
            f"- Correct answers: {self.correct_answers} ({self.accuracy * 100:.1f}%)\n"
            f"- Precision: {self.avg_precision:.4f}\n"
            f"- Recall: {self.avg_recall:.4f}\n"
            f"- F1 Score: {self.f1_score:.4f}\n"
            f"- Time: {format_time(self.time_used)}"
        )