from fastapi import FastAPI
from pydantic import BaseModel
from server.support_env import OpenSupportOpsEnv
from models import SupportAction

app = FastAPI()

env = OpenSupportOpsEnv()


@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: SupportAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state().model_dump()