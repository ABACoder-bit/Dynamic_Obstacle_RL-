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

# APF Parameters
K_ATTRACTIVE = 1.0
K_REPULSIVE = 4.0
REPULSIVE_RANGE = 3

# Local minima escape parameters
STUCK_THRESHOLD = 5
USE_RANDOM_KICK = True
USE_VIRTUAL_OBSTACLE = True
VIRTUAL_OBS_DURATION = 4


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


def euclidean_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def random_obstacles():
    obstacles = []

    while len(obstacles) < NUM_OBSTACLES:
        pos = (
            random.randint(0, GRID_SIZE - 1),
            random.randint(0, GRID_SIZE - 1)
        )

        if pos != START and pos != GOAL and pos not in [o['pos'] for o in obstacles]:
            speed = random.choice([1, 2])
            target = (
                random.randint(0, GRID_SIZE - 1),
                random.randint(0, GRID_SIZE - 1)
            )

            obstacles.append({
                'pos': pos,
                'speed': speed,
                'target': target
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
        for _ in range(obs['speed']):
            if random.random() < 0.7:
                new_pos = move_toward_target(obs['pos'], obs['target'])
            else:
                action = random.randint(0, 3)
                new_pos = move(obs['pos'], action)

            if new_pos == GOAL:
                continue

            obs['pos'] = new_pos

            if manhattan_distance(obs['pos'], obs['target']) == 0:
                obs['target'] = (
                    random.randint(0, GRID_SIZE - 1),
                    random.randint(0, GRID_SIZE - 1)
                )

    return obstacles


def obstacle_positions(obstacles):
    return [o['pos'] for o in obstacles]


def safe_action(robot, action, obstacles):
    next_pos = move(robot, action)
    return next_pos not in obstacle_positions(obstacles)


# -------------------------
# Q-Learning Policy
# -------------------------
def q_learning_action(robot, obstacles):
    state_idx = state_to_index(robot)
    sorted_actions = np.argsort(Q[state_idx])[::-1]

    for a in sorted_actions:
        if safe_action(robot, a, obstacles):
            return a

    return random.randint(0, 3)


# -------------------------
# APF Policy (Corrected)
# -------------------------
def attractive_potential(position):
    # Calculate scalar potential energy: U_att = 0.5 * k * distance^2
    dist = euclidean_distance(position, GOAL)
    return 0.5 * K_ATTRACTIVE * (dist ** 2)


def repulsive_potential(position, obstacles):
    # Calculate scalar potential energy: U_rep = 0.5 * k * (1/d - 1/d0)^2
    u_rep_total = 0

    for obs in obstacles:
        obs_pos = obs['pos']
        dist = euclidean_distance(position, obs_pos)

        if dist == 0:
            dist = 0.1  # Prevent division by zero

        if dist <= REPULSIVE_RANGE:
            strength = 0.5 * K_REPULSIVE * ((1 / dist) - (1 / REPULSIVE_RANGE)) ** 2
            u_rep_total += strength

    return u_rep_total


def apf_action(robot, obstacles, recent_positions, virtual_obstacle=None):
    best_action = None
    best_potential = float('inf') # We want to MINIMIZE potential

    # Detect local minima / stuck behavior
    stuck = False
    if len(recent_positions) >= STUCK_THRESHOLD:
        last_positions = recent_positions[-STUCK_THRESHOLD:]
        if len(set(last_positions)) <= 2:
            stuck = True

    # Random kick strategy
    if stuck and USE_RANDOM_KICK:
        valid_actions = []
        for a in range(4):
            if safe_action(robot, a, obstacles):
                valid_actions.append(a)

        if valid_actions:
            return random.choice(valid_actions), robot, VIRTUAL_OBS_DURATION

    # Virtual obstacle strategy
    effective_obstacles = list(obstacles)
    if virtual_obstacle is not None and USE_VIRTUAL_OBSTACLE:
        effective_obstacles.append({'pos': virtual_obstacle, 'speed': 0, 'target': virtual_obstacle})

    if stuck and USE_VIRTUAL_OBSTACLE and virtual_obstacle is None:
        virtual_obstacle = robot

    # Evaluate each possible next action
    for action in range(4):
        if not safe_action(robot, action, effective_obstacles):
            continue

        next_pos = move(robot, action)

        # Calculate scalar potential energy at the next position
        u_att = attractive_potential(next_pos)
        u_rep = repulsive_potential(next_pos, effective_obstacles)
        total_u = u_att + u_rep

        # We want the cell with the LOWEST potential energy
        if total_u < best_potential:
            best_potential = total_u
            best_action = action

    if best_action is None:
        return random.randint(0, 3), virtual_obstacle, VIRTUAL_OBS_DURATION

    return best_action, virtual_obstacle, VIRTUAL_OBS_DURATION if virtual_obstacle else 0


# -------------------------
# Runner
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
        recent_positions = []
        virtual_obstacle = None
        virtual_timer = 0
        result = 'TIMEOUT'

        for step in range(1, MAX_STEPS + 1):
            if agent_type == 'q_learning':
                action = q_learning_action(robot, obstacles)
            else:
                action, virtual_obstacle, virtual_timer = apf_action(
                    robot,
                    obstacles,
                    recent_positions,
                    virtual_obstacle
                )

            new_robot = move(robot, action)
            obstacles = move_obstacles(obstacles)
            obs_positions = obstacle_positions(obstacles)

            if new_robot in obs_positions:
                crash += 1
                result = 'CRASHED'
                steps_log.append(step)
                break

            if new_robot == GOAL:
                success += 1
                result = 'REACHED GOAL'
                steps_log.append(step)
                break

            recent_positions.append(new_robot)

            if virtual_timer > 0:
                virtual_timer -= 1
            else:
                virtual_obstacle = None

            visited_positions.append(new_robot)
            if visited_positions.count(new_robot) > 4:
                loop += 1
                result = 'LOOP'
                steps_log.append(step)
                break

            robot = new_robot

            if step == MAX_STEPS:
                timeout += 1
                steps_log.append(step)

        print(f"Episode {episode:03d} | {result:<13} | Steps: {steps_log[-1]}")

    return {
        'success': success,
        'crash': crash,
        'loop': loop,
        'timeout': timeout,
        'steps': steps_log
    }


# -------------------------
# Run Both Agents
# -------------------------
q_results = run_agent('q_learning')
apf_results = run_agent('apf')


# -------------------------
# Final Comparison
# -------------------------
print("\n================ FINAL COMPARISON ================")
print("\nQ-LEARNING RESULTS")
print(q_results)

print("\nAPF RESULTS")
print(apf_results)

print("\nSuccess Rate Comparison")
print(f"Q-Learning Success Rate : {q_results['success']}%")
print(f"APF Success Rate        : {apf_results['success']}%")


# -------------------------
# Comparison Graph
# -------------------------
labels = ['Reached Goal', 'Crashed', 'Loop', 'Timeout']

q_values = [
    q_results['success'],
    q_results['crash'],
    q_results['loop'],
    q_results['timeout']
]

apf_values = [
    apf_results['success'],
    apf_results['crash'],
    apf_results['loop'],
    apf_results['timeout']
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, q_values, width, label='Q-Learning')
plt.bar(x + width / 2, apf_values, width, label='APF')

plt.xticks(x, labels)
plt.ylabel('Number of Episodes')
plt.title('Q-Learning vs APF Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
