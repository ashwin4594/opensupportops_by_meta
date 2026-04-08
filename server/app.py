from fastapi import FastAPI
import uvicorn

from server.support_env import OpenSupportOpsEnv
from models import SupportAction

# Create FastAPI app
app = FastAPI(
    title="OpenSupportOps API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize environment
env = OpenSupportOpsEnv("easy_refund.json")


# Root endpoint (health check)
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


# Take a step
@app.post("/step")
def step(action: SupportAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


# Get state
@app.get("/state")
def state():
    return env.state()


# ✅ REQUIRED for OpenEnv validation
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# ✅ REQUIRED entry point
if __name__ == "__main__":
    main()