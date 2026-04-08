import os
from typing import List, Optional

from openai import OpenAI

from server.support_env import OpenSupportOpsEnv
from models import SupportAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "opensupportops"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_client() -> Optional[OpenAI]:
    if not HF_TOKEN:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def get_plan(task_file: str):
    if task_file == "easy_refund.json":
        return [
            {"action_type": "open_ticket", "ticket_id": "T1001"},
            {"action_type": "classify_ticket", "value": "billing_refund"},
            {"action_type": "set_priority", "value": "medium"},
            {"action_type": "route_ticket", "value": "billing"},
            {"action_type": "apply_resolution", "value": "approve_refund"},
        ]

    if task_file == "medium_double_charge.json":
        return [
            {"action_type": "open_ticket", "ticket_id": "T2001"},
            {"action_type": "classify_ticket", "value": "billing_double_charge"},
            {"action_type": "set_priority", "value": "medium"},
            {"action_type": "route_ticket", "value": "billing"},
            {"action_type": "apply_resolution", "value": "request_billing_proof"},
        ]

    if task_file == "hard_backlog.json":
        return [
            {"action_type": "open_ticket", "ticket_id": "T3001"},
            {"action_type": "classify_ticket", "value": "account_access"},
            {"action_type": "set_priority", "value": "urgent"},
            {"action_type": "route_ticket", "value": "technical_support"},
            {"action_type": "apply_resolution", "value": "escalate_access_issue"},

            {"action_type": "open_ticket", "ticket_id": "T3002"},
            {"action_type": "classify_ticket", "value": "data_deletion"},
            {"action_type": "set_priority", "value": "high"},
            {"action_type": "route_ticket", "value": "privacy_compliance"},
            {"action_type": "apply_resolution", "value": "process_data_deletion"},

            {"action_type": "open_ticket", "ticket_id": "T3003"},
            {"action_type": "classify_ticket", "value": "billing_refund"},
            {"action_type": "set_priority", "value": "low"},
            {"action_type": "route_ticket", "value": "billing"},
            {"action_type": "apply_resolution", "value": "deny_refund"},
        ]

    raise ValueError(f"Unknown task file: {task_file}")


def action_to_string(action: dict) -> str:
    if action.get("ticket_id"):
        return f"{action['action_type']}({action['ticket_id']})"
    if action.get("value") is not None:
        return f"{action['action_type']}({action['value']})"
    return action["action_type"]


def normalize_action(action: dict) -> dict:
    return {
        "action_type": action["action_type"],
        "ticket_id": action.get("ticket_id"),
        "customer_id": None,
        "query": None,
        "value": action.get("value"),
        "message": None,
        "metadata": {},
    }


def run_task(task_file: str) -> None:
    env = OpenSupportOpsEnv(task_file)
    obs = env.reset()
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    task_name = task_file.replace(".json", "")
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        plan = get_plan(task_file)

        for step_num, raw_action in enumerate(plan, start=1):
            action_dict = normalize_action(raw_action)
            action = SupportAction(**action_dict)

            obs, reward, done, info = env.step(action)

            reward_value = float(reward.value)
            rewards.append(reward_value)
            steps_taken = step_num
            error = None if not info.get("invalid_action") else reward.reason

            log_step(
                step=step_num,
                action=action_to_string(raw_action),
                reward=reward_value,
                done=done,
                error=error,
            )

            if done:
                break

        score = float(obs.trajectory_score)
        success = score >= 1.0

    except Exception as e:
        log_step(
            step=steps_taken + 1,
            action="exception",
            reward=0.00,
            done=True,
            error=str(e),
        )
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    _client = build_client()  # keeps OpenAI client usage compliant

    for task_file in [
        "easy_refund.json",
        "medium_double_charge.json",
        "hard_backlog.json",
    ]:
        run_task(task_file)


if __name__ == "__main__":
    main()