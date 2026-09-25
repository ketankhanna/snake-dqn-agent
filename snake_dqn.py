# Import libraries
import pygame
import random
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Game Constants
GRID_SIZE = 16  # Rows and columns of the grid
CELL_SIZE = 40  # Size of each grid cell in pixels
WINDOW_SIZE = GRID_SIZE * CELL_SIZE  # Total window size
INITIAL_FPS = 8  # Starting frames per second

# Learning Parameters
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
ALPHA = 0.3  # Learning rate for neural network
GAMMA = 0.95  # Discount factor for future rewards
EPSILON = 0.85  # Initial exploration rate
EPSILON_DECAY = 0.995  # Epsilon decay factor
MIN_EPSILON = 0.125  # Minimum exploration rate
BATCH_SIZE = 64 # Number of samples per training batch
MEMORY_SIZE = 5000  # Replay buffer size
TARGET_UPDATE = 4  # Episodes after which to update the target network
FOOD_PER_LEVEL = 1  # Food needed per level
MAX_LEVEL = 8  # Maximum level (game is winnable here)
MAX_FOOD = 4  # Total number of food items to eat across all levels before the game ends

# Global variables for kryptonite logic
failure_count = {}  # Track failures for each cell
failure_threshold = 2  # Failures before marking as kryptonite
boundary_penalty_memory = set()
grid_memory = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 0: unvisited, 1: explored, -1: dangerous

# Store the number of hits per dangerous position (boundary hits)
boundary_hit_count = {}  # Tracks how many times the boundary was hit

# Store which directions led to boundaries
dangerous_directions = {}  # Tracks directions that lead to boundary (e.g., up, down, left, right)

def build_model():
    """Builds an enhanced deep Q-network for predicting Q-values."""
    model = Sequential([
        Flatten(input_shape=(GRID_SIZE, GRID_SIZE)),
        Dense(256, activation='relu'),  
        Dense(256, activation='relu'), 
        Dense(128, activation='relu'),  
        Dense(len(ACTIONS), activation='linear')  # Output layer for Q-values
    ])
    model.compile(optimizer=Adam(learning_rate=ALPHA), loss='mse')
    return model

# Initialize models and memory
model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())
memory = deque(maxlen=MEMORY_SIZE)

def reset_game():
    """Resets the game state for a new episode. Initializes the snake at the center and places food randomly."""
    snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]  # Place snake in the center
    food = place_food(snake, (-1, -1))  # Pass a dummy last_food_position for the initial reset
    direction = (0, 1)  # Default direction (moving right)
    last_direction = direction
    level = 1

    # Removed print statement
    return snake, food, direction, last_direction, level


def place_food(snake, last_food_position):
    """Places food on a random safe cell, avoiding the snake's body, danger zones,ensuring a minimum distance from the last food position, and staying away from boundaries."""
    min_distance = max(2, GRID_SIZE // 2 - len(snake) // 4)  # Adaptive distance
    max_attempts = 100
    boundary_buffer = 2  # Minimum distance from the boundary

    attempts = 0
    while attempts < max_attempts:
        new_food = (
            random.randint(boundary_buffer, GRID_SIZE - 1 - boundary_buffer),
            random.randint(boundary_buffer, GRID_SIZE - 1 - boundary_buffer)
        )
        manhattan_distance = abs(new_food[0] - last_food_position[0]) + abs(new_food[1] - last_food_position[1])
        if new_food not in snake and grid_memory[new_food[0]][new_food[1]] != -1 and manhattan_distance >= min_distance:
            return new_food
        attempts += 1

    # Fallback in case no suitable position was found
    while True:
        fallback_food = (
            random.randint(boundary_buffer, GRID_SIZE - 1 - boundary_buffer),
            random.randint(boundary_buffer, GRID_SIZE - 1 - boundary_buffer)
        )
        if fallback_food not in snake and grid_memory[fallback_food[0]][fallback_food[1]] != -1:
            return fallback_food

def move_snake(snake, direction, last_direction):
    """Moves the snake while preventing reversal into its tail."""
    if direction == (-last_direction[0], -last_direction[1]):
        direction = last_direction

    head = snake[-1]
    new_head = (head[0] + direction[0], head[1] + direction[1])
    return new_head, direction

def check_collision(snake, new_head):
    """Checks for collisions with walls or itself."""
    return (new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE or
            new_head in snake)

def get_grid_state(snake, food):
    """Returns a grid representation of the current game state."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for segment in snake:
        grid[segment] = 1  # Snake's body
    grid[food] = 2  # Food
    return grid

def check_collision_type(snake, new_head, food):
    """Determines the type of collision: 'food', 'boundary', or 'tail'."""
    if new_head == food:
        return "food"
    elif new_head[0] < 0 or new_head[0] >= GRID_SIZE or new_head[1] < 0 or new_head[1] >= GRID_SIZE:
        return "boundary"
    elif new_head in snake[:-1]:  # Exclude the tail
        return "tail"
    return None

def choose_action(state, epsilon, snake):
    """Chooses an action using an epsilon-greedy policy. Prioritizes unexplored cells and avoids revisits if possible. Avoids dangerous cells (kryptonite) from grid memory."""
    if np.random.rand() < epsilon:
        valid_actions = []
        for action in range(len(ACTIONS)):
            head = snake[-1]
            predicted_new_head = (head[0] + ACTIONS[action][0], head[1] + ACTIONS[action][1])

            if 0 <= predicted_new_head[0] < GRID_SIZE and 0 <= predicted_new_head[1] < GRID_SIZE:
                if grid_memory[predicted_new_head[0]][predicted_new_head[1]] != -1:
                    valid_actions.append(action)

        if valid_actions:
            action = random.choice(valid_actions)
        else:
            action = random.randint(0, len(ACTIONS) - 1)
    else:
        q_values = model.predict(state[np.newaxis, ...])
        action = np.argmax(q_values[0])

        head = snake[-1]
        predicted_new_head = (head[0] + ACTIONS[action][0], head[1] + ACTIONS[action][1])
        if 0 <= predicted_new_head[0] < GRID_SIZE and 0 <= predicted_new_head[1] < GRID_SIZE:
            if grid_memory[predicted_new_head[0]][predicted_new_head[1]] == -1:
                return choose_action(state, epsilon, snake)

    return action

def replay_and_train():
    """Trains the model using experiences from the replay buffer. Penalizes revisiting danger cells and rewards safe exploration."""
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    for state, action, reward, next_state, done in batch:
        target = model.predict(state[np.newaxis, ...])

        if done:
            target[0][action] = reward
        else:
            next_q = np.max(target_model.predict(next_state[np.newaxis, ...]))
            target[0][action] = reward + GAMMA * next_q

        model.fit(state[np.newaxis, ...], target, epochs=10, verbose=0)  

def update_target_network():
    """Updates the target network to sync with the main model."""
    target_model.set_weights(model.get_weights())

def draw_game(snake, food, screen):
    jungle_texture = pygame.Surface((CELL_SIZE, CELL_SIZE))
    jungle_texture.fill((34, 77, 34))

    snake_head_texture = pygame.Surface((CELL_SIZE // 1.5, CELL_SIZE // 1.5), pygame.SRCALPHA)
    pygame.draw.ellipse(snake_head_texture, (144, 238, 144), snake_head_texture.get_rect())

    eye_radius = CELL_SIZE // 12
    eye_x_offset = CELL_SIZE // 8
    eye_y_offset = CELL_SIZE // 6
    pupil_radius = eye_radius // 2

    pygame.draw.circle(snake_head_texture, (255, 255, 255), (eye_x_offset, eye_y_offset), eye_radius)
    pygame.draw.circle(snake_head_texture, (0, 0, 0), (eye_x_offset, eye_y_offset), pupil_radius) 

    pygame.draw.circle(snake_head_texture, (255, 255, 255), (snake_head_texture.get_width() - eye_x_offset, eye_y_offset), eye_radius)  
    pygame.draw.circle(snake_head_texture, (0, 0, 0), (snake_head_texture.get_width() - eye_x_offset, eye_y_offset), pupil_radius)  

    snake_tongue_texture = pygame.Surface((CELL_SIZE // 6, CELL_SIZE // 12), pygame.SRCALPHA)
    pygame.draw.rect(snake_tongue_texture, (255, 0, 0), pygame.Rect(0, 0, CELL_SIZE // 6, CELL_SIZE // 12))

    snake_body_texture = pygame.Surface((CELL_SIZE // 1.5, CELL_SIZE // 1.5), pygame.SRCALPHA)
    pygame.draw.rect(snake_body_texture, (85, 107, 47), snake_body_texture.get_rect())

    for x in range(0, WINDOW_SIZE, jungle_texture.get_width()):
        for y in range(0, WINDOW_SIZE, jungle_texture.get_height()):
            screen.blit(jungle_texture, (x, y))

    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        for y in range(0, WINDOW_SIZE, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (139, 69, 19), rect, 2)
            pygame.draw.rect(screen, (85, 107, 47), rect, 1)

    border_thickness = 5
    pygame.draw.rect(screen, (139, 115, 85), pygame.Rect(0, 0, WINDOW_SIZE, WINDOW_SIZE), border_thickness)

    for i, segment in enumerate(snake):
        segment_rect = pygame.Rect(
            segment[1] * CELL_SIZE + CELL_SIZE // 6,
            segment[0] * CELL_SIZE + CELL_SIZE // 6,
            CELL_SIZE // 1.5,
            CELL_SIZE // 1.5
        )
        if i == len(snake) - 1:
            screen.blit(snake_head_texture, segment_rect)
            tongue_rect = pygame.Rect(
                segment[1] * CELL_SIZE + CELL_SIZE // 3 - CELL_SIZE // 12,
                segment[0] * CELL_SIZE + CELL_SIZE // 2 + CELL_SIZE // 8,
                CELL_SIZE // 6,
                CELL_SIZE // 12
            )
            screen.blit(snake_tongue_texture, tongue_rect)
        else:
            screen.blit(snake_body_texture, segment_rect)

    food_rect = pygame.Rect(food[1] * CELL_SIZE, food[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    glow_color = (255, 69, 0) if pygame.time.get_ticks() // 300 % 2 == 0 else (255, 165, 0)
    pygame.draw.ellipse(screen, glow_color, food_rect.inflate(-12, -10))
    pygame.draw.ellipse(screen, (255, 0, 0), food_rect.inflate(-16, -14))
    pygame.draw.rect(
        screen,
        (34, 139, 34),
        pygame.Rect(
            food_rect.centerx - CELL_SIZE // 35,
            food_rect.y - CELL_SIZE // 12,
            CELL_SIZE // 15,
            CELL_SIZE // 7
        )
    )

    pygame.display.flip()





import time

def run_snake_game():
    """Main game loop for the Snake game with reinforcement learning. Handles danger zones, prevents backtracking, ensures safe food placement and maintains distance constraints for new food."""
    global EPSILON
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Snake Game with RL")
    clock = pygame.time.Clock()

    max_episodes = 25000
    max_levels = 8
    total_food_eaten = 0
    collected_food = 0
    level = 1
    game_over = False

    level_fps = {i: 8 + (i - 1) * 2 for i in range(1, max_levels + 1)}
    INITIAL_FPS = level_fps[level]

    last_food_position = (-1, -1)

    # Metrics Tracking
    start_time = time.time()
    end_time = start_time + 3600  # Run for 1 hour
    episode_rewards = []  # Total rewards per episode
    levels_reached = []  # Levels reached per episode
    food_collected_per_episode = []  # Food collected per episode

    for episode in range(1, max_episodes + 1):
        INITIAL_FPS = level_fps[level]
        food_to_level_up = 1
        print(f"=== Episode {episode}/{max_episodes} - Level {level} (FPS: {INITIAL_FPS}, Food Needed: {food_to_level_up}) ===")

        if game_over:  # Reset game if the previous state ended in failure
            print("Game Over. Resetting game...")
            snake, food, direction, last_direction, _ = reset_game()  # Reset without touching level
            level = 1  # Explicitly reset the level to 1
            collected_food = 0  # Reset food counter
            game_over = False  # Reset the game over flag
            pygame.time.delay(1000)  # Optional delay for clarity
            continue
        else:  # Standard reset for a new episode
            snake, food, direction, last_direction, _ = reset_game()

        running = True
        initial_moves = 5
        episode_reward = 0  # Cumulative reward for the episode
        food_collected = 0  # Food collected in the current episode

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            reward = 0

            # Snake movement logic
            if initial_moves > 0:
                if direction == (0, 1):
                    direction = (1, 0)
                elif direction == (1, 0):
                    direction = (0, -1)
                elif direction == (0, -1):
                    direction = (-1, 0)
                elif direction == (-1, 0):
                    direction = (0, 1)
                initial_moves -= 1
            else:
                reverse_direction = (-last_direction[0], -last_direction[1])
                valid_actions = [
                    action_idx for action_idx, action in enumerate(ACTIONS)
                    if action != reverse_direction
                ]

                if np.random.rand() < EPSILON:
                    action_idx = random.choice(valid_actions)
                else:
                    q_values = model.predict(get_grid_state(snake, food)[np.newaxis, ...])
                    action_idx = np.argmax(q_values[0]) if np.argmax(q_values[0]) in valid_actions else random.choice(valid_actions)

                direction = ACTIONS[action_idx]

            # Calculate new position
            new_head, direction = move_snake(snake, direction, last_direction)
            last_direction = direction

            # Check collisions
            if not (0 <= new_head[0] < GRID_SIZE and 0 <= new_head[1] < GRID_SIZE):
                print(f"Game Over - Hit boundary at {new_head}. Ignoring out-of-bounds.")
                game_over = True  # Trigger Game Over
                break

            collision_type = check_collision_type(snake[:-1], new_head, food)
            if collision_type == "boundary":
                reward = -1000
                print(f"Game Over - Hit boundary at {new_head}. Marking as kryptonite.")
                grid_memory[new_head[0]][new_head[1]] = -1
                game_over = True  # Trigger Game Over
                break
            elif collision_type == "tail":
                reward = -1000
                print(f"Game Over - Hit tail at {new_head}.")
                game_over = True  # Trigger Game Over
                break
            elif collision_type == "food":
                reward = 5000
                collected_food += 1
                food_collected += 1
                total_food_eaten += 1
                food_remaining = max(0, food_to_level_up - collected_food)  # Prevent negatives
                print(f"Collected food! Remaining: {food_remaining}")
                last_food_position = food
                food = place_food(snake, last_food_position)

                # Level up when enough food is collected
                if collected_food >= food_to_level_up:
                    collected_food = 0
                    level += 1
                    print(f"Level up! Now on level {level}.")
                    if level > max_levels:
                        print(f"Maximum level {max_levels} reached. Game will continue.")

            else:
                snake.pop(0)

            # Update snake position
            snake.append(new_head)
            draw_game(snake, food, screen)

            clock.tick(INITIAL_FPS)
            episode_reward += reward

        # Track metrics for this episode
        episode_rewards.append(episode_reward)
        levels_reached.append(level)
        food_collected_per_episode.append(food_collected)

        # Update exploration rate
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        if episode % TARGET_UPDATE == 0:
            update_target_network()

        # Check if 1 hour has passed
        if time.time() > end_time:
            print("1-hour training complete!")
            break




run_snake_game()

