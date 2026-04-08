from server.support_env import OpenSupportOpsEnv
from models import SupportAction


def test_easy_task():
    env = OpenSupportOpsEnv("easy_refund.json")
    env.reset()
    env.step(SupportAction(action_type="open_ticket", ticket_id="T1001"))
    env.step(SupportAction(action_type="classify_ticket", value="billing_refund"))
    env.step(SupportAction(action_type="set_priority", value="medium"))
    env.step(SupportAction(action_type="route_ticket", value="billing"))
    obs, reward, done, info = env.step(
        SupportAction(action_type="apply_resolution", value="approve_refund")
    )
    assert obs.trajectory_score == 1.0


def test_medium_task():
    env = OpenSupportOpsEnv("medium_double_charge.json")
    env.reset()
    env.step(SupportAction(action_type="open_ticket", ticket_id="T2001"))
    env.step(SupportAction(action_type="classify_ticket", value="billing_double_charge"))
    env.step(SupportAction(action_type="set_priority", value="medium"))
    env.step(SupportAction(action_type="route_ticket", value="billing"))
    obs, reward, done, info = env.step(
        SupportAction(action_type="apply_resolution", value="request_billing_proof")
    )
    assert obs.trajectory_score == 1.0


def test_hard_task():
    env = OpenSupportOpsEnv("hard_backlog.json")
    env.reset()

    env.step(SupportAction(action_type="open_ticket", ticket_id="T3001"))
    env.step(SupportAction(action_type="classify_ticket", value="account_access"))
    env.step(SupportAction(action_type="set_priority", value="urgent"))
    env.step(SupportAction(action_type="route_ticket", value="technical_support"))
    env.step(SupportAction(action_type="apply_resolution", value="escalate_access_issue"))

    env.step(SupportAction(action_type="open_ticket", ticket_id="T3002"))
    env.step(SupportAction(action_type="classify_ticket", value="data_deletion"))
    env.step(SupportAction(action_type="set_priority", value="high"))
    env.step(SupportAction(action_type="route_ticket", value="privacy_compliance"))
    env.step(SupportAction(action_type="apply_resolution", value="process_data_deletion"))

    env.step(SupportAction(action_type="open_ticket", ticket_id="T3003"))
    env.step(SupportAction(action_type="classify_ticket", value="billing_refund"))
    env.step(SupportAction(action_type="set_priority", value="low"))
    env.step(SupportAction(action_type="route_ticket", value="billing"))
    obs, reward, done, info = env.step(
        SupportAction(action_type="apply_resolution", value="deny_refund")
    )

    assert obs.trajectory_score == 1.0

def test_wrong_action_not_full_score():
    env = OpenSupportOpsEnv("easy_refund.json")
    env.reset()

    env.step(SupportAction(action_type="open_ticket", ticket_id="T1001"))
    env.step(SupportAction(action_type="classify_ticket", value="wrong_label"))
    env.step(SupportAction(action_type="set_priority", value="medium"))
    env.step(SupportAction(action_type="route_ticket", value="billing"))

    obs, reward, done, info = env.step(
        SupportAction(action_type="apply_resolution", value="approve_refund")
    )

    assert obs.trajectory_score == 0.75