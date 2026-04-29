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
Q = np.load("q_table_improved.npy")


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


def distance(a, b):
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

            if distance(obs["pos"], obs["target"]) == 0:
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
# Testing Loop (100 Episodes)
# -------------------------
print("\n===== TESTING TRAINED AGENT (100 EPISODES) =====\n")

success_count = 0
crash_count = 0
loop_count = 0
max_step_count = 0
steps_per_episode = []
episode_result = []

for episode in range(1, TOTAL_EPISODES + 1):
    robot = START
    obstacles = random_obstacles()
    visited_positions = []
    result = "MAX_STEPS"

    for step in range(1, MAX_STEPS + 1):
        state_idx = state_to_index(robot)

        sorted_actions = np.argsort(Q[state_idx])[::-1]

        action = None
        for a in sorted_actions:
            if safe_action(robot, a, obstacles):
                action = a
                break

        if action is None:
            action = random.randint(0, 3)

        new_robot = move(robot, action)

        obstacles = move_obstacles(obstacles)
        obs_positions = obstacle_positions(obstacles)

        if new_robot in obs_positions:
            crash_count += 1
            result = "CRASHED"
            steps_per_episode.append(step)
            break

        if new_robot == GOAL:
            success_count += 1
            result = "REACHED GOAL"
            steps_per_episode.append(step)
            break

        visited_positions.append(new_robot)
        if visited_positions.count(new_robot) > 4:
            loop_count += 1
            result = "STUCK IN LOOP"
            steps_per_episode.append(step)
            break

        robot = new_robot

        if step == MAX_STEPS:
            max_step_count += 1
            steps_per_episode.append(step)

    episode_result.append(result)

    print(f"Episode {episode:03d} | Result: {result:<15} | Steps: {steps_per_episode[-1]}")


# -------------------------
# Final Summary
# -------------------------
print("\n===== FINAL SUMMARY =====")
print(f"Total Episodes      : {TOTAL_EPISODES}")
print(f"Reached Goal        : {success_count}")
print(f"Crashed             : {crash_count}")
print(f"Stuck in Loop       : {loop_count}")
print(f"Max Steps Reached   : {max_step_count}")


# -------------------------
# Graph
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, TOTAL_EPISODES + 1), steps_per_episode, marker='o')
plt.title("Steps Taken Per Episode")
plt.xlabel("Episode Number")
plt.ylabel("Steps Taken")
plt.grid(True)
plt.tight_layout()
plt.show()
