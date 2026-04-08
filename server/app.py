from fastapi import FastAPI
from server.support_env import OpenSupportOpsEnv
from models import SupportAction

app = FastAPI(
    title="OpenSupportOps API",
    docs_url="/docs",     # ✅ ensures docs are available
    redoc_url="/redoc"    # optional
)

# Initialize environment (default task)
env = OpenSupportOpsEnv("easy_refund.json")


# Root endpoint (for health check)
@app.get("/")
def root():
    return {"message": "OpenSupportOps API is running"}


# Reset environment
@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": obs,
        "message": "Environment reset successful"
    }


# Take a step in environment
@app.post("/step")
def step(action: SupportAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


# Get full state
@app.get("/state")
def state():
    return env.state()