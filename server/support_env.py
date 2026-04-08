import json
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

from models import (
    RewardModel,
    SupportAction,
    SupportObservation,
    SupportState,
    TicketDetail,
    TicketSummary,
)

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


class OpenSupportOpsEnv:
    def __init__(self, task_file: str = "easy_refund.json"):
        self.task_file = task_file
        self.task_data: Dict[str, Any] = {}
        self._state: SupportState | None = None

    def _load_task(self) -> Dict[str, Any]:
        with open(TASKS_DIR / self.task_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def reset(self) -> SupportObservation:
        self.task_data = self._load_task()

        tickets = {
            t["ticket_id"]: TicketDetail(**t)
            for t in self.task_data["visible"]["tickets"]
        }

        self._state = SupportState(
            task_id=self.task_data["task_id"],
            episode_id=str(uuid.uuid4()),
            step_count=0,
            max_steps=self.task_data.get("max_steps", 10),
            done=False,
            active_ticket_id=None,
            tickets=tickets,
            customers=self.task_data["visible"].get("customers", {}),
            policy_snippets=self.task_data["visible"].get("policy_snippets", []),
            internal_notes={k: [] for k in tickets.keys()},
            draft_replies={},
            resolutions={},
            audit_log=[],
            hidden_score=0.0,
        )

        return self._build_observation("reset_done")

    def state(self) -> SupportState:
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state

    def step(
        self, action: SupportAction
    ) -> Tuple[SupportObservation, RewardModel, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() first")

        self._state.step_count += 1
        info: Dict[str, Any] = {"invalid_action": False}

        self._state.audit_log.append(
            {
                "step": self._state.step_count,
                "action": action.model_dump(),
            }
        )

        reward = RewardModel(value=0.0, reason="step")

        if action.action_type == "open_ticket":
            if action.ticket_id and action.ticket_id in self._state.tickets:
                self._state.active_ticket_id = action.ticket_id
                reward = RewardModel(value=0.1, reason="opened_ticket")
            else:
                reward = RewardModel(value=-0.1, reason="invalid_ticket")
                info["invalid_action"] = True

        elif action.action_type == "view_customer":
            if self._state.active_ticket_id is None:
                reward = RewardModel(value=-0.05, reason="no_active_ticket")
                info["invalid_action"] = True
            else:
                reward = RewardModel(value=0.02, reason="customer_viewed")

        elif action.action_type == "search_policy":
            reward = RewardModel(value=0.02, reason="policy_searched")

        elif action.action_type == "classify_ticket":
            if self._state.active_ticket_id is None:
                reward = RewardModel(value=-0.05, reason="no_active_ticket")
                info["invalid_action"] = True
            else:
                ticket = self._state.tickets[self._state.active_ticket_id]
                ticket.category = action.value
                if ticket.ticket_id in self.task_data["gold"]["classification"]:
                    expected = self.task_data["gold"]["classification"][ticket.ticket_id]
                    if action.value == expected:
                        reward = RewardModel(value=0.1, reason="classification_correct")
                    else:
                        reward = RewardModel(value=-0.02, reason="classification_wrong")
                else:
                    reward = RewardModel(value=0.05, reason="ticket_classified")

        elif action.action_type == "set_priority":
            if self._state.active_ticket_id is None:
                reward = RewardModel(value=-0.05, reason="no_active_ticket")
                info["invalid_action"] = True
            elif action.value not in {"low", "medium", "high", "urgent"}:
                reward = RewardModel(value=-0.05, reason="invalid_priority")
                info["invalid_action"] = True
            else:
                ticket = self._state.tickets[self._state.active_ticket_id]
                ticket.priority = action.value
                if ticket.ticket_id in self.task_data["gold"]["priority"]:
                    expected = self.task_data["gold"]["priority"][ticket.ticket_id]
                    if action.value == expected:
                        reward = RewardModel(value=0.1, reason="priority_correct")
                    else:
                        reward = RewardModel(value=-0.02, reason="priority_wrong")
                else:
                    reward = RewardModel(value=0.05, reason="priority_updated")

        elif action.action_type == "route_ticket":
            if self._state.active_ticket_id is None:
                reward = RewardModel(value=-0.05, reason="no_active_ticket")
                info["invalid_action"] = True
            else:
                ticket = self._state.tickets[self._state.active_ticket_id]
                ticket.assigned_team = action.value
                if ticket.ticket_id in self.task_data["gold"]["routing"]:
                    expected = self.task_data["gold"]["routing"][ticket.ticket_id]
                    if action.value == expected:
                        reward = RewardModel(value=0.15, reason="routing_correct")
                    else:
                        reward = RewardModel(value=-0.03, reason="routing_wrong")
                else:
                    reward = RewardModel(value=0.05, reason="ticket_routed")

        elif action.action_type == "request_more_info":
            if self._state.active_ticket_id is None:
                reward = RewardModel(value=-0.05, reason="no_active_ticket")
                info["invalid_action"] = True
            else:
                ticket = self._state.tickets[self._state.active_ticket_id]
                ticket.status = "pending"
                reward = RewardModel(value=0.08, reason="more_info_requested")

        elif action.action_type == "draft_reply":
            if self._state.active_ticket_id is None:
                reward = RewardModel(value=-0.05, reason="no_active_ticket")
                info["invalid_action"] = True
            else:
                self._state.draft_replies[self._state.active_ticket_id] = action.message or ""
                reward = RewardModel(value=0.08, reason="reply_drafted")

        elif action.action_type == "add_internal_note":
            if self._state.active_ticket_id is None:
                reward = RewardModel(value=-0.05, reason="no_active_ticket")
                info["invalid_action"] = True
            else:
                ticket_id = self._state.active_ticket_id
                self._state.internal_notes.setdefault(ticket_id, []).append(action.message or "")
                reward = RewardModel(value=0.02, reason="internal_note_added")

        elif action.action_type == "apply_resolution":
            if self._state.active_ticket_id is None:
                reward = RewardModel(value=-0.05, reason="no_active_ticket")
                info["invalid_action"] = True
            else:
                ticket_id = self._state.active_ticket_id
                self._state.resolutions[ticket_id] = action.value or ""
                if ticket_id in self.task_data["gold"]["resolution"]:
                    expected = self.task_data["gold"]["resolution"][ticket_id]
                    if action.value == expected:
                        reward = RewardModel(value=0.2, reason="resolution_correct")
                    else:
                        reward = RewardModel(value=-0.05, reason="resolution_wrong")
                else:
                    reward = RewardModel(value=0.1, reason="resolution_applied")

        elif action.action_type == "close_ticket":
            if self._state.active_ticket_id is None:
                reward = RewardModel(value=-0.05, reason="no_active_ticket")
                info["invalid_action"] = True
            else:
                ticket = self._state.tickets[self._state.active_ticket_id]
                ticket.status = "closed"
                reward = RewardModel(value=0.1, reason="ticket_closed")

        elif action.action_type == "escalate":
            if self._state.active_ticket_id is None:
                reward = RewardModel(value=-0.05, reason="no_active_ticket")
                info["invalid_action"] = True
            else:
                ticket = self._state.tickets[self._state.active_ticket_id]
                ticket.assigned_team = action.value or "escalations"
                ticket.status = "pending"
                reward = RewardModel(value=0.05, reason="ticket_escalated")

        else:
            reward = RewardModel(value=-0.1, reason="unsupported_action")
            info["invalid_action"] = True

        self._update_progress_score()

        done = self._state.step_count >= self._state.max_steps
        if done:
            self._state.done = True

        obs = self._build_observation(reward.reason)
        return obs, reward, done, info

    def _update_progress_score(self) -> None:
        if self._state is None:
            return

        gold = self.task_data.get("gold", {})
        score = 0.0

        classification = gold.get("classification", {})
        priority = gold.get("priority", {})
        routing = gold.get("routing", {})
        resolution = gold.get("resolution", {})

        for tid, expected in classification.items():
            if tid in self._state.tickets and self._state.tickets[tid].category == expected:
                score += 0.25

        for tid, expected in priority.items():
            if tid in self._state.tickets and self._state.tickets[tid].priority == expected:
                score += 0.25

        for tid, expected in routing.items():
            if tid in self._state.tickets and self._state.tickets[tid].assigned_team == expected:
                score += 0.25

        for tid, expected in resolution.items():
            if self._state.resolutions.get(tid) == expected:
                score += 0.25

        self._state.hidden_score = round(min(score, 1.0), 2)

    def _build_observation(self, last_action_result: str) -> SupportObservation:
        s = self._state
        assert s is not None

        selected = s.tickets.get(s.active_ticket_id) if s.active_ticket_id else None

        return SupportObservation(
            screen="ticket" if selected else "inbox",
            visible_tickets=[
                TicketSummary(
                    ticket_id=t.ticket_id,
                    subject=t.subject,
                    customer_tier=t.metadata.get("customer_tier", "free"),
                    priority=t.priority,
                    status=t.status,
                    category=t.category,
                )
                for t in s.tickets.values()
            ],
            selected_ticket=selected,
            visible_policy_hits=s.policy_snippets[:3],
            last_action_result=last_action_result,
            valid_next_actions=[
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
            ],
            remaining_steps=max(0, s.max_steps - s.step_count),
            trajectory_score=s.hidden_score,
        )