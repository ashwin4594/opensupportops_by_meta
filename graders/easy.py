def grade_easy(task_data, state):
    gold = task_data["gold"]

    score = 0
    total = 4

    tickets = state.tickets
    resolutions = state.resolutions

    for tid, expected in gold["classification"].items():
        if tickets[tid].category == expected:
            score += 1

    for tid, expected in gold["priority"].items():
        if tickets[tid].priority == expected:
            score += 1

    for tid, expected in gold["routing"].items():
        if tickets[tid].assigned_team == expected:
            score += 1

    for tid, expected in gold["resolution"].items():
        if resolutions.get(tid) == expected:
            score += 1

    return round(score / total, 2)