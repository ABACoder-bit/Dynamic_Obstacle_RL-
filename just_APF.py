import numpy as np
import random
import matplotlib.pyplot as plt

# =====================================
# PURE APF FOR DYNAMIC OBSTACLE AVOIDANCE
# =====================================

GRID_SIZE = 12
START = (0, 0)
GOAL = (11, 11)
NUM_OBSTACLES = 6
MAX_STEPS = 150
TOTAL_EPISODES = 100

# APF PARAMETERS
K_ATTRACTIVE = 1.2
K_REPULSIVE = 8.0
REPULSIVE_RANGE = 3

# ESCAPE FROM LOCAL MINIMA
STUCK_THRESHOLD = 6
RANDOM_KICK_PROB = 0.8


# =====================================
# BASIC HELPERS
# =====================================

def move(position, action):
    x, y = position

    if action == 0:  # UP
        x = max(0, x - 1)
    elif action == 1:  # DOWN
        x = min(GRID_SIZE - 1, x + 1)
    elif action == 2:  # LEFT
        y = max(0, y - 1)
    elif action == 3:  # RIGHT
        y = min(GRID_SIZE - 1, y + 1)

    return (x, y)


def euclidean(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# =====================================
# OBSTACLE GENERATION
# =====================================

def random_obstacles():
    obstacles = []

    while len(obstacles) < NUM_OBSTACLES:
        pos = (
            random.randint(0, GRID_SIZE - 1),
            random.randint(0, GRID_SIZE - 1)
        )

        if pos != START and pos != GOAL and pos not in [o['pos'] for o in obstacles]:
            obstacles.append({
                'pos': pos,
                'speed': random.choice([1, 2]),
                'target': (
                    random.randint(0, GRID_SIZE - 1),
                    random.randint(0, GRID_SIZE - 1)
                )
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
                new_pos = move(obs['pos'], random.randint(0, 3))

            if new_pos != GOAL:
                obs['pos'] = new_pos

            if obs['pos'] == obs['target']:
                obs['target'] = (
                    random.randint(0, GRID_SIZE - 1),
                    random.randint(0, GRID_SIZE - 1)
                )

    return obstacles


def obstacle_positions(obstacles):
    return [o['pos'] for o in obstacles]


# =====================================
# APF FORCE FUNCTIONS
# =====================================

def attractive_force(pos):
    fx = K_ATTRACTIVE * (GOAL[0] - pos[0])
    fy = K_ATTRACTIVE * (GOAL[1] - pos[1])
    return fx, fy


def repulsive_force(pos, obstacles):
    fx_total = 0
    fy_total = 0

    for obs in obstacles:
        ox, oy = obs['pos']
        d = euclidean(pos, (ox, oy))

        if d == 0:
            d = 0.1

        if d <= REPULSIVE_RANGE:
            strength = K_REPULSIVE * ((1 / d) - (1 / REPULSIVE_RANGE)) / (d ** 2)
            fx_total += strength * (pos[0] - ox)
            fy_total += strength * (pos[1] - oy)

    return fx_total, fy_total


# =====================================
# LOCAL MINIMA DETECTION
# =====================================

def is_stuck(history):
    if len(history) < STUCK_THRESHOLD:
        return False

    recent = history[-STUCK_THRESHOLD:]

    # if robot keeps repeating same 2–3 cells
    return len(set(recent)) <= 3


# =====================================
# APF ACTION SELECTION
# =====================================

def apf_action(robot, obstacles, history):
    # Random kick to escape local minima
    if is_stuck(history):
        if random.random() < RANDOM_KICK_PROB:
            valid = []
            for a in range(4):
                next_pos = move(robot, a)
                if next_pos not in obstacle_positions(obstacles):
                    valid.append(a)

            if valid:
                return random.choice(valid)

    best_action = None
    best_score = -float('inf')

    for action in range(4):
        next_pos = move(robot, action)

        if next_pos in obstacle_positions(obstacles):
            continue

        fa_x, fa_y = attractive_force(next_pos)
        fr_x, fr_y = repulsive_force(next_pos, obstacles)

        total_fx = fa_x + fr_x
        total_fy = fa_y + fr_y

        # prefer progress to goal + obstacle avoidance
        force_score = (total_fx ** 2 + total_fy ** 2) ** 0.5
        goal_bonus = -euclidean(next_pos, GOAL)

        score = force_score + goal_bonus

        if score > best_score:
            best_score = score
            best_action = action

    if best_action is None:
        return random.randint(0, 3)

    return best_action


# =====================================
# RUN TEST
# =====================================

def run_apf():
    success = 0
    crash = 0
    loop = 0
    timeout = 0
    steps_log = []

    print("\n===== PURE APF TEST STARTED =====\n")

    for episode in range(1, TOTAL_EPISODES + 1):
        robot = START
        obstacles = random_obstacles()
        history = []
        result = "TIMEOUT"

        for step in range(1, MAX_STEPS + 1):
            action = apf_action(robot, obstacles, history)
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

            history.append(new_robot)

            if history.count(new_robot) > 6:
                loop += 1
                result = "LOOP"
                steps_log.append(step)
                break

            robot = new_robot

            if step == MAX_STEPS:
                timeout += 1
                steps_log.append(step)

        print(f"Episode {episode:03d} | {result:<13} | Steps: {steps_log[-1]}")

    print("\n=========== FINAL RESULTS ===========")
    print(f"Reached Goal : {success}")
    print(f"Crashed      : {crash}")
    print(f"Loop         : {loop}")
    print(f"Timeout      : {timeout}")

    plt.figure(figsize=(8, 5))
    plt.bar(
        ["Goal", "Crash", "Loop", "Timeout"],
        [success, crash, loop, timeout]
    )
    plt.title("Pure APF Performance")
    plt.ylabel("Episodes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_apf()
