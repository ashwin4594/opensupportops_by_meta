from server.support_env import OpenSupportOpsEnv
from models import SupportAction


def run_easy():
    print("=== EASY TASK TEST ===")
    env = OpenSupportOpsEnv("easy_refund.json")

    obs = env.reset()
    print("RESET:", obs.screen)

    obs, reward, done, info = env.step(
        SupportAction(action_type="open_ticket", ticket_id="T1001")
    )
    print("STEP 1:", obs.screen, reward)

    obs, reward, done, info = env.step(
        SupportAction(action_type="classify_ticket", value="billing_refund")
    )
    print("STEP 2 SCORE:", obs.trajectory_score)

    obs, reward, done, info = env.step(
        SupportAction(action_type="set_priority", value="medium")
    )
    print("STEP 3 SCORE:", obs.trajectory_score)

    obs, reward, done, info = env.step(
        SupportAction(action_type="route_ticket", value="billing")
    )
    print("STEP 4 SCORE:", obs.trajectory_score)

    obs, reward, done, info = env.step(
        SupportAction(action_type="apply_resolution", value="approve_refund")
    )
    print("EASY FINAL SCORE:", obs.trajectory_score)
    print()


def run_medium():
    print("=== MEDIUM TASK TEST ===")
    env = OpenSupportOpsEnv("medium_double_charge.json")

    obs = env.reset()
    print("RESET:", obs.screen)

    obs, reward, done, info = env.step(
        SupportAction(action_type="open_ticket", ticket_id="T2001")
    )

    obs, reward, done, info = env.step(
        SupportAction(action_type="classify_ticket", value="billing_double_charge")
    )

    obs, reward, done, info = env.step(
        SupportAction(action_type="set_priority", value="medium")
    )

    obs, reward, done, info = env.step(
        SupportAction(action_type="route_ticket", value="billing")
    )

    obs, reward, done, info = env.step(
        SupportAction(action_type="apply_resolution", value="request_billing_proof")
    )

    print("MEDIUM FINAL SCORE:", obs.trajectory_score)
    print()


def run_hard():
    print("=== HARD TASK TEST ===")
    env = OpenSupportOpsEnv("hard_backlog.json")

    obs = env.reset()
    print("RESET:", obs.screen)

    # Ticket 1
    obs, reward, done, info = env.step(
        SupportAction(action_type="open_ticket", ticket_id="T3001")
    )
    obs, reward, done, info = env.step(
        SupportAction(action_type="classify_ticket", value="account_access")
    )
    obs, reward, done, info = env.step(
        SupportAction(action_type="set_priority", value="urgent")
    )
    obs, reward, done, info = env.step(
        SupportAction(action_type="route_ticket", value="technical_support")
    )
    obs, reward, done, info = env.step(
        SupportAction(action_type="apply_resolution", value="escalate_access_issue")
    )

    # Ticket 2
    obs, reward, done, info = env.step(
        SupportAction(action_type="open_ticket", ticket_id="T3002")
    )
    obs, reward, done, info = env.step(
        SupportAction(action_type="classify_ticket", value="data_deletion")
    )
    obs, reward, done, info = env.step(
        SupportAction(action_type="set_priority", value="high")
    )
    obs, reward, done, info = env.step(
        SupportAction(action_type="route_ticket", value="privacy_compliance")
    )
    obs, reward, done, info = env.step(
        SupportAction(action_type="apply_resolution", value="process_data_deletion")
    )

    # Ticket 3
    obs, reward, done, info = env.step(
        SupportAction(action_type="open_ticket", ticket_id="T3003")
    )
    obs, reward, done, info = env.step(
        SupportAction(action_type="classify_ticket", value="billing_refund")
    )
    obs, reward, done, info = env.step(
        SupportAction(action_type="set_priority", value="low")
    )
    obs, reward, done, info = env.step(
        SupportAction(action_type="route_ticket", value="billing")
    )
    obs, reward, done, info = env.step(
        SupportAction(action_type="apply_resolution", value="deny_refund")
    )

    print("HARD FINAL SCORE:", obs.trajectory_score)
    print()


if __name__ == "__main__":
    run_easy()
    run_medium()
    run_hard()