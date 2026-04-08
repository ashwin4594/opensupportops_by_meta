from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TicketSummary(BaseModel):
    ticket_id: str
    subject: str
    customer_tier: Literal["free", "pro", "enterprise"]
    priority: Literal["low", "medium", "high", "urgent"]
    status: Literal["open", "pending", "resolved", "closed"]
    category: Optional[str] = None


class TicketDetail(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_id: str
    status: Literal["open", "pending", "resolved", "closed"] = "open"
    priority: Literal["low", "medium", "high", "urgent"] = "low"
    category: Optional[str] = None
    assigned_team: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class SupportAction(BaseModel):
    action_type: Literal[
        "open_ticket",
        "view_customer",
        "search_policy",
        "classify_ticket",
        "set_priority",
        "route_ticket",
        "request_more_info",
        "draft_reply",
        "add_internal_note",
        "apply_resolution",
        "close_ticket",
        "escalate",
    ]
    ticket_id: Optional[str] = None
    customer_id: Optional[str] = None
    query: Optional[str] = None
    value: Optional[str] = None
    message: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class SupportObservation(BaseModel):
    screen: Literal["inbox", "ticket", "customer", "policy", "final"]
    visible_tickets: List[TicketSummary] = Field(default_factory=list)
    selected_ticket: Optional[TicketDetail] = None
    visible_policy_hits: List[str] = Field(default_factory=list)
    last_action_result: str = ""
    valid_next_actions: List[str] = Field(default_factory=list)
    remaining_steps: int
    trajectory_score: float = 0.0


class RewardModel(BaseModel):
    value: float
    reason: str


class SupportState(BaseModel):
    task_id: str
    episode_id: str
    step_count: int
    max_steps: int
    done: bool = False
    active_ticket_id: Optional[str] = None
    tickets: Dict[str, TicketDetail] = Field(default_factory=dict)
    customers: Dict[str, dict] = Field(default_factory=dict)
    policy_snippets: List[str] = Field(default_factory=list)
    draft_replies: Dict[str, str] = Field(default_factory=dict)
    internal_notes: Dict[str, List[str]] = Field(default_factory=dict)
    resolutions: Dict[str, str] = Field(default_factory=dict)
    audit_log: List[dict] = Field(default_factory=list)
    hidden_score: float = 0.0