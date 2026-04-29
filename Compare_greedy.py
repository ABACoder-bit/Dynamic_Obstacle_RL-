import numpy as np
import random
import matplotlib.pyplot as plt

# -------------------------
# Environment Settings
# -------------------------
GRID_SIZE = 12
NUM_OBSTACLES = 6
START = (0, 0)
GOAL = (11, 11)
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

TOTAL_EPISODES = 1000
MAX_STEPS = 150

# Load trained Q-table
Q = np.load("q_table_v3.npy")


# -------------------------
# Helper Functions
# -------------------------
def state_to_index(state):
    return state[0] * GRID_SIZE + state[1]


def move(position, action):
    x, y = position

    if action == 0:  # UP
        x = max(x - 1, 0)
    elif action == 1:  # DOWN
        x = min(x + 1, GRID_SIZE - 1)
    elif action == 2:  # LEFT
        y = max(y - 1, 0)
    elif action == 3:  # RIGHT
        y = min(y + 1, GRID_SIZE - 1)

    return (x, y)


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def random_obstacles():
    obstacles = []

    while len(obstacles) < NUM_OBSTACLES:
        pos = (
            random.randint(0, GRID_SIZE - 1),
            random.randint(0, GRID_SIZE - 1)
        )

        if pos != START and pos != GOAL and pos not in [o["pos"] for o in obstacles]:
            speed = random.choice([1, 2])
            target = (
                random.randint(0, GRID_SIZE - 1),
                random.randint(0, GRID_SIZE - 1)
            )

            obstacles.append({
                "pos": pos,
                "speed": speed,
                "target": target
            })

    return obstacles


def move_toward_target(pos, target):
    x, y = pos
    tx, ty = target

    if x < tx:
        x += 1
    elif x > tx:
        x -= 1
    elif y < ty:
        y += 1
    elif y > ty:
        y -= 1

    return (x, y)


def move_obstacles(obstacles):
    for obs in obstacles:
        for _ in range(obs["speed"]):
            if random.random() < 0.7:
                new_pos = move_toward_target(obs["pos"], obs["target"])
            else:
                action = random.randint(0, 3)
                new_pos = move(obs["pos"], action)

            if new_pos == GOAL:
                continue

            obs["pos"] = new_pos

            if manhattan_distance(obs["pos"], obs["target"]) == 0:
                obs["target"] = (
                    random.randint(0, GRID_SIZE - 1),
                    random.randint(0, GRID_SIZE - 1)
                )

    return obstacles


def obstacle_positions(obstacles):
    return [o["pos"] for o in obstacles]


def safe_action(robot, action, obstacles):
    next_pos = move(robot, action)
    return next_pos not in obstacle_positions(obstacles)


# -------------------------
# Policy Functions
# -------------------------
def q_learning_action(robot, obstacles):
    state_idx = state_to_index(robot)
    sorted_actions = np.argsort(Q[state_idx])[::-1]

    for a in sorted_actions:
        if safe_action(robot, a, obstacles):
            return a

    return random.randint(0, 3)


def greedy_action(robot, obstacles):
    best_action = None
    best_distance = float("inf")

    for action in range(4):
        if not safe_action(robot, action, obstacles):
            continue

        next_pos = move(robot, action)
        dist = manhattan_distance(next_pos, GOAL)

        if dist < best_distance:
            best_distance = dist
            best_action = action

    if best_action is None:
        return random.randint(0, 3)

    return best_action


# -------------------------
# Single Agent Runner
# -------------------------
def run_agent(agent_type):
    success = 0
    crash = 0
    loop = 0
    timeout = 0
    steps_log = []

    print(f"\n===== RUNNING {agent_type.upper()} AGENT =====\n")

    for episode in range(1, TOTAL_EPISODES + 1):
        robot = START
        obstacles = random_obstacles()
        visited_positions = []
        result = "TIMEOUT"

        for step in range(1, MAX_STEPS + 1):

            if agent_type == "q_learning":
                action = q_learning_action(robot, obstacles)
            else:
                action = greedy_action(robot, obstacles)

            new_robot = move(robot, action)
            obstacles = move_obstacles(obstacles)
            obs_positions = obstacle_positions(obstacles)

            if new_robot in obs_positions:
                crash += 1
                result = "CRASHED"
                steps_log.append(step)
                break

            if new_robot == GOAL:
                success += 1
                result = "REACHED GOAL"
                steps_log.append(step)
                break

            visited_positions.append(new_robot)
            if visited_positions.count(new_robot) > 4:
                loop += 1
                result = "LOOP"
                steps_log.append(step)
                break

            robot = new_robot

            if step == MAX_STEPS:
                timeout += 1
                steps_log.append(step)

        print(f"Episode {episode:03d} | {result:<13} | Steps: {steps_log[-1]}")

    return {
        "success": success,
        "crash": crash,
        "loop": loop,
        "timeout": timeout,
        "steps": steps_log
    }


# -------------------------
# Run Both Agents
# -------------------------
q_results = run_agent("q_learning")
g_results = run_agent("greedy")


# -------------------------
# Final Comparison Summary
# -------------------------
print("\n================ FINAL COMPARISON ================")
print("\nQ-LEARNING RESULTS")
print(q_results)

print("\nGREEDY RESULTS")
print(g_results)

print("\nSuccess Rate Comparison")
print(f"Q-Learning Success Rate : {q_results['success']}%")
print(f"Greedy Success Rate     : {g_results['success']}%")


# -------------------------
# Comparison Graph
# -------------------------
labels = ["Reached Goal", "Crashed", "Loop", "Timeout"]
q_values = [
    q_results["success"],
    q_results["crash"],
    q_results["loop"],
    q_results["timeout"]
]

g_values = [
    g_results["success"],
    g_results["crash"],
    g_results["loop"],
    g_results["timeout"]
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, q_values, width, label="Q-Learning")
plt.bar(x + width / 2, g_values, width, label="Greedy")

plt.xticks(x, labels)
plt.ylabel("Number of Episodes")
plt.title("Q-Learning vs Greedy Baseline Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
