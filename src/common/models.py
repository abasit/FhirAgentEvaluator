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
    question_id: Optional[str] = None
    question: Optional[str] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None

    # Debug info
    iterations: int = 0
    tools_used: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_fhir_ids: dict[str, list] = Field(default_factory=dict)

    # Populated after evaluation
    true_answer: Optional[str] = None
    correct: Optional[int] = None
    true_fhir_ids: dict[str, list] = Field(default_factory=dict)
    precision: Optional[float] = None
    recall: Optional[float] = None


class FHIRAgentBenchResult(BaseModel):
    """Aggregated benchmark results with per-task details."""
    total_tasks: int
    correct_answers: int
    accuracy: float
    avg_precision: float
    avg_recall: float
    f1_score: float
    time_used: float
    task_results: list[TaskResult]