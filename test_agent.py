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
            speed = random.choice([1, 2])  # Variable obstacle speed
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

            # 70% goal-directed movement
            if random.random() < 0.7:
                new_pos = move_toward_target(obs["pos"], obs["target"])
            else:
                action = random.randint(0, 3)
                new_pos = move(obs["pos"], action)

            # Prevent obstacle from occupying goal
            if new_pos == GOAL:
                continue

            obs["pos"] = new_pos

            # If target reached → assign new target
            if distance(obs["pos"], obs["target"]) == 0:
                obs["target"] = (
                    random.randint(0, GRID_SIZE - 1),
                    random.randint(0, GRID_SIZE - 1)
                )

    return obstacles


def obstacle_positions(obstacles):
    return [o["pos"] for o in obstacles]


def safe_action(robot, action, obstacles):
    """
    Robot 'vision':
    it avoids moving directly into visible obstacles
    """
    next_pos = move(robot, action)
    return next_pos not in obstacle_positions(obstacles)


# -------------------------
# Visualization
# -------------------------
def visualize(robot, obstacles, step):
    plt.clf()

    # Draw grid
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            plt.scatter(
                y,
                GRID_SIZE - 1 - x,
                marker="s",
                s=300
            )

    # Goal
    plt.scatter(
        GOAL[1],
        GRID_SIZE - 1 - GOAL[0],
        marker="*",
        s=700,
        label="Goal"
    )

    # Obstacles
    for obs in obstacles:
        plt.scatter(
            obs["pos"][1],
            GRID_SIZE - 1 - obs["pos"][0],
            marker="X",
            s=500,
            label="Obstacle"
        )

    # Robot
    plt.scatter(
        robot[1],
        GRID_SIZE - 1 - robot[0],
        marker="o",
        s=700,
        label="Robot"
    )

    plt.title(f"Dynamic Obstacle Avoidance (Step {step})")
    plt.xticks([])
    plt.yticks([])
    plt.pause(0.35)


# -------------------------
# Testing Loop
# -------------------------
print("\nTesting trained agent...\n")

robot = START
obstacles = random_obstacles()

visited_positions = []

plt.figure(figsize=(7, 7))

for step in range(150):

    state_idx = state_to_index(robot)

    # Choose best safe action using robot vision
    sorted_actions = np.argsort(Q[state_idx])[::-1]

    action = None
    for a in sorted_actions:
        if safe_action(robot, a, obstacles):
            action = a
            break

    # fallback if all blocked
    if action is None:
        action = random.randint(0, 3)

    new_robot = move(robot, action)

    # Move dynamic obstacles
    obstacles = move_obstacles(obstacles)
    obs_positions = obstacle_positions(obstacles)

    # Logging
    print(f"\nStep {step}")
    print("Robot:", robot, "→", new_robot)
    print("Action:", ACTIONS[action])
    print("Obstacle Positions:", obs_positions)
    print("-" * 50)

    # Visualize
    visualize(new_robot, obstacles, step)

    # Crash detection
    if new_robot in obs_positions:
        print("❌ Robot crashed into obstacle!")
        break

    # Goal reached
    if new_robot == GOAL:
        print("✅ Robot reached goal successfully!")
        break

    # Loop detection
    visited_positions.append(new_robot)
    if visited_positions.count(new_robot) > 4:
        print("⚠️ Robot stuck in loop at:", new_robot)
        break

    robot = new_robot

plt.show()