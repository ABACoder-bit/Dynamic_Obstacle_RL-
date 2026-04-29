import numpy as np
import random

# -------------------------
# Environment Settings
# -------------------------
GRID_SIZE = 12
NUM_OBSTACLES = 6

START = (0, 0)
GOAL = (11, 11)

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

alpha = 0.1
gamma = 0.9
epsilon = 0.3
episodes = 5000

Q = np.zeros((GRID_SIZE * GRID_SIZE, len(ACTIONS)))


def state_to_index(state):
    return state[0] * GRID_SIZE + state[1]


def move(position, action):
    x, y = position

    if action == 0:
        x = max(x - 1, 0)
    elif action == 1:
        x = min(x + 1, GRID_SIZE - 1)
    elif action == 2:
        y = max(y - 1, 0)
    elif action == 3:
        y = min(y + 1, GRID_SIZE - 1)

    return (x, y)


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def random_obstacles():
    obs = []
    while len(obs) < NUM_OBSTACLES:
        pos = (random.randint(0, GRID_SIZE - 1),
               random.randint(0, GRID_SIZE - 1))

        if pos != START and pos != GOAL and pos not in obs:
            speed = random.choice([1, 2])
            target = (random.randint(0, GRID_SIZE - 1),
                      random.randint(0, GRID_SIZE - 1))
            obs.append({
                "pos": pos,
                "speed": speed,
                "target": target
            })
    return obs


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


print("\nTraining Started...\n")

for ep in range(episodes):
    robot = START
    obstacles = random_obstacles()
    visited = set()

    for step in range(250):
        state_idx = state_to_index(robot)

        if random.random() < epsilon:
            candidate_actions = list(range(4))
            random.shuffle(candidate_actions)
        else:
            candidate_actions = np.argsort(Q[state_idx])[::-1]

        action = None
        for a in candidate_actions:
            if safe_action(robot, a, obstacles):
                action = a
                break

        if action is None:
            action = random.randint(0, 3)

        new_robot = move(robot, action)

        obstacles = move_obstacles(obstacles)
        obs_positions = obstacle_positions(obstacles)

        reward = -1
        done = False

        if new_robot in obs_positions:
            reward = -100
            done = True

        elif new_robot == GOAL:
            reward = 300
            done = True

        else:
            if distance(new_robot, GOAL) < distance(robot, GOAL):
                reward += 4

            if new_robot in visited:
                reward -= 6

        visited.add(new_robot)

        new_state_idx = state_to_index(new_robot)

        Q[state_idx, action] += alpha * (
            reward + gamma * np.max(Q[new_state_idx]) - Q[state_idx, action]
        )

        robot = new_robot

        if done:
            break

    if ep % 500 == 0:
        print(f"Episode {ep} completed")

np.save("q_table.npy", Q)
print("\nTraining Finished. Q-table saved.\n")