from server.support_env import OpenSupportOpsEnv
from models import SupportAction


def run_task(task_file, actions):
    env = OpenSupportOpsEnv(task_file)
    obs = env.reset()

    for action in actions:
        obs, reward, done, info = env.step(SupportAction(**action))

    return obs.trajectory_score


def main():
    easy_actions = [
        {"action_type": "open_ticket", "ticket_id": "T1001"},
        {"action_type": "classify_ticket", "value": "billing_refund"},
        {"action_type": "set_priority", "value": "medium"},
        {"action_type": "route_ticket", "value": "billing"},
        {"action_type": "apply_resolution", "value": "approve_refund"},
    ]

    medium_actions = [
        {"action_type": "open_ticket", "ticket_id": "T2001"},
        {"action_type": "classify_ticket", "value": "billing_double_charge"},
        {"action_type": "set_priority", "value": "medium"},
        {"action_type": "route_ticket", "value": "billing"},
        {"action_type": "apply_resolution", "value": "request_billing_proof"},
    ]

    hard_actions = [
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

    easy_score = run_task("easy_refund.json", easy_actions)
    medium_score = run_task("medium_double_charge.json", medium_actions)
    hard_score = run_task("hard_backlog.json", hard_actions)

    avg_score = round((easy_score + medium_score + hard_score) / 3, 2)

    print("Easy Score:", easy_score)
    print("Medium Score:", medium_score)
    print("Hard Score:", hard_score)
    print("Average Score:", avg_score)


if __name__ == "__main__":
    main()