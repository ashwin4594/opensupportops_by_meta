---
title: OpenSupportOps
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# OpenSupportOps

## What this project is about

OpenSupportOps is a simulation environment that mimics how real customer support teams handle tickets.

Instead of solving simple one-step problems, this environment requires an agent to go through multiple steps like understanding the issue, setting priority, routing it correctly, and resolving it — similar to how real support agents work.

---

## Why this matters

Customer support is a real-world problem where decisions matter. Agents need to quickly understand issues, follow policies, and take the right actions.

This project is designed to simulate that workflow so AI agents can be trained and evaluated in a more practical and realistic setting.

---

## How it works

The environment follows a simple loop:

- `reset()` → starts a new task  
- `step(action)` → performs an action  
- `state()` → returns the full environment state  

---

## Actions supported

- open_ticket  
- classify_ticket  
- set_priority  
- route_ticket  
- apply_resolution  
- escalate  

---

## What the agent sees

Each observation includes:

- list of tickets  
- selected ticket details  
- customer information  
- policy snippets  
- valid next actions  
- current score  

---

## Scoring system

The score increases step by step when actions are correct:

- classification → +0.25  
- priority → +0.25  
- routing → +0.25  
- resolution → +0.25  

Final score ranges from **0.0 to 1.0**

---

## Tasks

### Easy  
Simple refund request.

### Medium  
Billing issue that requires additional reasoning.

### Hard  
Multiple tickets with different priorities and workflows.

---

## Baseline performance

- Easy: 1.0  
- Medium: 1.0  
- Hard: 1.0  
- Average: 1.0  

---

## How to run

### Install dependencies

pip install -r requirements.txt

### Start API

uvicorn server.app:app --reload

### Run tests

python -m pytest

### Run baseline

python -m baseline.run_baseline

---

## Docker

docker build -f server/Dockerfile -t opensupportops .
docker run -p 7860:7860 opensupportops

Open in browser:

http://127.0.0.1:7860/docs

---

## API Endpoints

- /reset  
- /step  
- /state  

---

## Key points

- Real-world customer support simulation  
- Multi-step decision making  
- Partial reward scoring (not just pass/fail)  
- Deterministic evaluation  
- Fully tested  
- Docker-ready  

---

## Final note

This project is designed to be simple to run, but realistic enough to represent actual support workflows. It can be used to train and evaluate AI agents on structured decision-making tasks.