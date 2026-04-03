from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentRequestOptions:
    temperature: float
    top_p: Optional[float]
    top_k: Optional[int]
    response_format: Optional[str]
    max_output_tokens: Optional[int]
    stop_sequences: list[str]
    finish_instruction: Optional[str]
    count_tokens: bool = False


@dataclass
class AgentTokenStats:
    # Tokens for current user message contribution in the current request (excludes system prompt).
    current_request_tokens: Optional[int]
    # Tokens for previous chat messages contribution in the current request (excludes system prompt).
    dialog_history_tokens: Optional[int]
    # Tokens produced by the model (completion tokens).
    response_model_tokens: Optional[int]
    # prompt_tokens as returned by the provider for the actual request.
    prompt_tokens_total: Optional[int]
    # completion_tokens as returned by the provider for the actual request.
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


@dataclass
class AgentResponse:
    answer: str
    raw_data: dict
    model: str
    latency_sec: float
    provider: str
    token_stats: Optional[dict] = None


@dataclass
class FactsState:
    facts: dict[str, str] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)


@dataclass
class BranchState:
    root_messages: list[dict[str, str]] = field(default_factory=list)
    checkpoint_messages: Optional[list[dict[str, str]]] = None
    branches: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    active_branch: Optional[str] = None


@dataclass
class TaskState:
    task: str = ""
    state: str = "PLANNING"
    paused: bool = False
    step: int = 0
    total: int = 0
    expected_action: str = ""
    plan: list[str] = field(default_factory=list)
    done: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ShortTermMemory:
    dialog_tail: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class WorkingMemory:
    current_task: TaskState = field(default_factory=TaskState)


@dataclass
class LongTermMemory:
    profile: dict[str, str] = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)
    knowledge: dict[str, str] = field(default_factory=dict)
    invariants: list[str] = field(default_factory=list)
