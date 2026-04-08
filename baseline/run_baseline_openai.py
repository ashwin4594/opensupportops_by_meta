import json
import os

from openai import OpenAI

from server.support_env import OpenSupportOpsEnv
from models import SupportAction


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_prompt(obs):
    return f"""
You are operating a customer support environment.

Choose exactly one next action in JSON.

Valid next actions:
{obs.valid_next_actions}

Current screen:
{obs.screen}

Visible tickets:
{obs.visible_tickets}

Selected ticket:
{obs.selected_ticket}

Visible policy hits:
{obs.visible_policy_hits}

Last action result:
{obs.last_action_result}

Remaining steps:
{obs.remaining_steps}

Trajectory score:
{obs.trajectory_score}

Return only JSON in this format:
{{
  "action_type": "open_ticket",
  "ticket_id": null,
  "customer_id": null,
  "query": null,
  "value": null,
  "message": null,
  "metadata": {{}}
}}
""".strip()


def ask_model(obs):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a precise support operations agent. Return JSON only."},
            {"role": "user", "content": build_prompt(obs)},
        ],
        temperature=0,
    )

    text = response.choices[0].message.content.strip()
    return json.loads(text)


def run_task(task_file):
    env = OpenSupportOpsEnv(task_file)
    obs = env.reset()
    done = False

    while not done and obs.remaining_steps > 0:
        action_dict = ask_model(obs)
        action = SupportAction(**action_dict)
        obs, reward, done, info = env.step(action)

    return obs.trajectory_score


def main():
    easy_score = run_task("easy_refund.json")
    medium_score = run_task("medium_double_charge.json")
    hard_score = run_task("hard_backlog.json")

    avg_score = round((easy_score + medium_score + hard_score) / 3, 2)

    print("Easy Score:", easy_score)
    print("Medium Score:", medium_score)
    print("Hard Score:", hard_score)
    print("Average Score:", avg_score)


if __name__ == "__main__":
    main()