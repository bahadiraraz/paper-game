import random
import sys

import numpy as np
import pygame
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model


class MapConfig(BaseModel):
    top_left_color: list = Field([255, 0, 0])
    top_right_color: list = Field([0, 255, 0])
    bottom_left_color: list = Field([0, 150, 255])
    bottom_right_color: list = Field([255, 215, 0])
    window_size: int = Field(1000, ge=1, le=1000)
    player_size: int = Field(10, ge=1, le=10)
    caption: str = Field("map")
    font_size: int = Field(20, ge=1, le=100)
    font_type: str = Field("arial")


pygame.init()
MapConfig = MapConfig()

grid_size = 1000
player_size = MapConfig.player_size

if grid_size % player_size != 0:
    grid_size += player_size - (grid_size % player_size)

grid = np.full(
    (grid_size, grid_size, 3),
    (255, 255, 255),
)
screen_size = grid_size * player_size
screen = pygame.display.set_mode((MapConfig.window_size, MapConfig.window_size))
pygame.display.set_caption(MapConfig.caption)

font = pygame.font.SysFont(MapConfig.font_type, MapConfig.font_size)

grid_size = 80
color_size = int(grid_size * 0.06)

for i in range(grid_size):
    for j in range(grid_size):
        if (i > color_size * 2 and i < color_size * 3) and (
            j > color_size * 2 and j < color_size * 3
        ):
            grid[i][j] = MapConfig.top_left_color
        elif (i > color_size * 6 and i < color_size * 7) and (
            j > color_size * 2 and j < color_size * 3
        ):
            grid[i][j] = MapConfig.top_right_color
        elif (i > color_size * 2 and i < color_size * 3) and (
            j > color_size * 6 and j < color_size * 7
        ):
            grid[i][j] = MapConfig.bottom_left_color
        elif (i > color_size * 6 and i < color_size * 7) and (
            j > color_size * 6 and j < color_size * 7
        ):
            grid[i][j] = MapConfig.bottom_right_color


player_size = 5
actions = [
    (x, y, x + player_size - 1, y + player_size - 1)
    for x in range(0, grid_size - player_size + 1, player_size)
    for y in range(0, grid_size - player_size + 1, player_size)
]


def get_color_name(color):
    if color == MapConfig.top_left_color:
        return "Red"
    if color == MapConfig.top_right_color:
        return "Green"
    if color == MapConfig.bottom_left_color:
        return "Blue"
    if color == MapConfig.bottom_right_color:
        return "Yellow"


running = True
dragging = False
start_pos = None
end_pos = None

colors = [
    np.array([255, 0, 0]),
    np.array([0, 255, 0]),
    np.array([0, 150, 255]),
    np.array([255, 215, 0]),
]


def get_state(grid, color):
    color_mask = (grid == color).all(axis=2)
    color_total_area = np.sum(color_mask)
    if color_total_area == 0:
        return [0, 0, 0]
    color_mean = np.mean(grid[color_mask], axis=0) / 255
    opponent_colors = [c for c in colors if not np.array_equal(c, color)]
    opponent_total_area = 0
    for opponent_color in opponent_colors:
        opponent_mask = (grid == opponent_color).all(axis=2)
        opponent_total_area += np.sum(opponent_mask)
    return np.array(
        [
            color_total_area / (100 * 100),
            color_mean[0],
            color_mean[1],
            color_mean[2],
            opponent_total_area / (100 * 100),
        ]
    )


def ai_function(color, grid):
    opponent_colors = [c for c in colors if not np.array_equal(c, color)]
    opponent_colors = [
        c
        for c in opponent_colors
        if np.any((grid == c).all(axis=2) & (grid == color).all(axis=2))
    ]
    while True:
        color_mask = (grid == color).all(axis=2)
        color_total_area = np.sum(color_mask)
        if color_total_area == 0:
            print(f"No {color} zone found.")
            continue

        if color_total_area > (grid.shape[0] * grid.shape[1]) / 2:
            min_opponent_area = float("inf")
            min_opponent_pos = None
            for opponent_color in opponent_colors:
                opponent_mask = (grid == opponent_color).all(axis=2)
                opponent_area = np.sum(opponent_mask)
                if opponent_area < min_opponent_area:
                    opponent_indices = np.argwhere(opponent_mask)
                    color_indices = np.argwhere(color_mask)
                    distances = np.sqrt(
                        np.sum(
                            (opponent_indices[:, np.newaxis, :] - color_indices) ** 2,
                            axis=-1,
                        )
                    )
                    min_distance = np.min(distances)
                    if np.isfinite(min_distance):
                        min_opponent_area = opponent_area
                        min_opponent_pos = opponent_indices[np.argmin(distances)]
            if min_opponent_pos is not None:
                start_x, start_y = min_opponent_pos
                max_area = min_opponent_area
                print(f"{color} zone is too big. Starting to eat smaller areas.")
            else:
                start_pos = np.argwhere(color_mask)[np.random.randint(color_total_area)]
                start_x, start_y = start_pos
                max_area = color_total_area
        else:
            start_pos = np.argwhere(color_mask)[np.random.randint(color_total_area)]
            start_x, start_y = start_pos
            max_area = color_total_area

        end_x = np.clip(start_x + np.random.randint(1, max_area + 1), 0, 99)
        end_y = np.clip(start_y + np.random.randint(1, max_area + 1), 0, 99)

        start_x, end_x = np.sort([start_x, end_x])
        start_y, end_y = np.sort([start_y, end_y])
        selected_area = (end_x - start_x + 1) * (end_y - start_y + 1)

        if selected_area > color_total_area:
            print("Selected area is greater than the maximum available area.")
            continue
        elif selected_area <= 0:
            print("Selected area cannot be zero.")
            continue

        grid_mask = np.zeros_like(grid)
        grid_mask[start_x : end_x + 1, start_y : end_y + 1] = 1
        white_mask = (grid == (255, 255, 255)).all(axis=2)[..., np.newaxis]
        opponent_mask = np.zeros_like(grid_mask)
        for opponent_color in opponent_colors:
            opponent_mask |= (grid == opponent_color).all(axis=2)[..., np.newaxis]
            if (grid_mask & opponent_mask).any():
                print("Selected area overlaps with opponent color.")
                continue
            elif (grid_mask & (grid != color).all(axis=2)[..., np.newaxis]).any():
                print("Selected area overlaps with existing color.")
                continue
            elif (grid_mask & white_mask).sum() < selected_area:
                print("Selected area overlaps with existing color.")
                continue
        reward = get_reward(grid, color, start_x, start_y, end_x, end_y)

        if selected_area > color_total_area:
            print("Selected area is greater than the color's total area.")
            reward -= 0.5
        elif (grid_mask & (grid != color).all(axis=2)[..., np.newaxis]).any():
            print("Selected area contains opponent color.")
            reward += 0.5
            if selected_area == color_total_area:
                reward += 1.0
        elif (grid_mask & white_mask).sum() == selected_area:
            print("Selected area contains only white spots.")
            reward += 0.2
        else:
            print("Selected area is invalid.")
            reward -= 0.2

        probability = random.random()
        if probability <= 0.999:
            grid[start_x : end_x + 1, start_y : end_y + 1] = color
            print("Area bought.")
            return color_total_area, selected_area, reward
        else:
            grid_mask = np.zeros_like(grid)
            color_indices = np.where((grid == color).all(axis=2))
            color_indices = list(
                zip(color_indices[0][:selected_area], color_indices[1][:selected_area])
            )
            for x, y in color_indices:
                grid[x, y] = (255, 255, 255)
            print("Area not bought.")
            return color_total_area, 0, 0


def select_action(model, state, actions):
    state = np.asarray(state)
    state = state.reshape(1, -1)
    try:
        q_values = model.predict(state)
        action = np.argmax(q_values[0])
        return actions[action]
    except ValueError:
        print("ValueError: Cannot select action.")
        return None


def get_reward(grid, color, start_x, start_y, end_x, end_y):
    start_grid_x, start_grid_y = start_x // 10, start_y // 10
    end_grid_x, end_grid_y = end_x // 10, end_y // 10
    selected_area = (end_grid_x - start_grid_x + 1) * (end_grid_y - start_grid_y + 1)
    total_color_square = np.sum((grid == color).all(axis=2))
    white_mask = (grid == (255, 255, 255)).all(axis=2)

    if selected_area > total_color_square:
        reward = -1.0
    elif (white_mask[start_x : end_x + 1, start_y : end_y + 1]).all():
        reward = -0.5
    else:
        reward = 0.2
        captured_opponent = False
        has_white = (white_mask[start_x : end_x + 1, start_y : end_y + 1]).any()
        has_color = (
            (grid[start_x : end_x + 1, start_y : end_y + 1] != color).all(axis=2)
            & ~white_mask[start_x : end_x + 1, start_y : end_y + 1]
        ).any()
        if has_white and has_color:
            reward += 0.1
        for opponent_color in [c for c in colors if not np.array_equal(c, color)]:
            opponent_mask = (grid == opponent_color).all(axis=2)
            opponent_total_square = np.sum(opponent_mask)
            if (
                opponent_total_square > 0
                and (opponent_mask[start_x : end_x + 1, start_y : end_y + 1]).any()
            ):
                captured_opponent = True
                reward += 0.5
                break
        if not captured_opponent and not has_white and not has_color:
            reward -= 0.1
        if selected_area == 0:
            reward -= 0.5
        if (grid != color).all(axis=2).any():
            reward -= 0.1

    return reward


def train_model(model, grid, color, actions, episodes):
    color = np.array(color)
    for episode in range(episodes):
        state = get_state(grid, color)
        action = select_action(model, state, actions)
        if action is not None:
            start_x, start_y = action[0] * 10, action[1] * 10
            end_x, end_y = action[2] * 10 + 9, action[3] * 10 + 9
            ai_function(color, grid)
            reward = get_reward(grid, color, start_x, start_y, end_x, end_y)
            next_state = get_state(grid, color)
            model.fit(
                np.array(state).reshape(1, -1),
                np.array(
                    [
                        reward
                        + 0.99
                        * np.max(model.predict(np.array(next_state).reshape(1, -1)))
                    ]
                ),
                epochs=1,
                verbose=0,
            )

        if np.sum((grid == color).all(axis=2)) == 7000:
            print(f"{color} has captured the whole map!")
            break


def train_models():
    models = [Sequential() for _ in range(4)]
    for model in models:
        model.add(Dense(24, input_dim=5, activation="relu"))  # Updated input_dim to 5
        model.add(Dense(48, activation="relu"))
        model.add(Dense(3, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

    for color, model in zip(colors, models):
        train_model(model, grid, np.array(color), actions, episodes=1000)
    for i, model in enumerate(models):
        model.save(f"model_{i}.h5")


models = [load_model(f"model_{i}.h5") for i in range(4)]
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        def set_color(grid, pos):
            color = grid[pos[0] // 10, pos[1] // 10]
            return color

        if event.type == pygame.MOUSEBUTTONDOWN:
            for model in models:
                for color in colors:
                    state = get_state(grid, color)
                    action = select_action(model, state, actions)
                    if action is not None:
                        start_x, start_y = action[0] * 10, action[1] * 10
                        end_x, end_y = action[2] * 10 + 9, action[3] * 10 + 9
                        ai_function(color, grid)
            start_color = set_color(grid, event.pos)

            start_pos = event.pos
            dragging = True

        if event.type == pygame.MOUSEBUTTONUP:
            end_pos = event.pos
            dragging = False

            start_x, start_y = start_pos
            end_x, end_y = end_pos

            if start_x > end_x:
                start_x, end_x = end_x, start_x

            if start_y > end_y:
                start_y, end_y = end_y, start_y

            start_grid_x, start_grid_y = start_x // 10, start_y // 10
            end_grid_x, end_grid_y = end_x // 10, end_y // 10
            selected_area = (end_grid_x - start_grid_x + 1) * (
                end_grid_y - start_grid_y + 1
            )
            total_color_square = np.sum((grid == start_color).all(axis=2))

            if selected_area < total_color_square:
                probability = random.random()
                if probability <= 0.99:
                    grid[
                        start_grid_x : end_grid_x + 1, start_grid_y : end_grid_y + 1
                    ] = start_color
                else:
                    color_indices = np.where((grid == start_color).all(axis=2))
                    color_indices = list(
                        zip(
                            color_indices[0][:selected_area],
                            color_indices[1][:selected_area],
                        )
                    )
                    for x, y in color_indices:
                        grid[x][y] = (255, 255, 255)
            else:
                print("Selected area cannot be greater than total color square.")

        for x in range(100):
            for y in range(100):
                if event.type == pygame.MOUSEMOTION and dragging:
                    end_pos = event.pos
                    start_x, start_y = start_pos
                    end_x, end_y = end_pos
                    if start_x > end_x:
                        start_x, end_x = end_x, start_x
                    if start_y > end_y:
                        start_y, end_y = end_y, start_y

                    dragged_area = (start_x, start_y, end_x - start_x, end_y - start_y)
                    pygame.draw.rect(screen, (0, 0, 255), dragged_area, 1)
                    text = font.render(
                        f"{int(dragged_area[2] / 7)} x {int(dragged_area[3] / 7)}",
                        True,
                        (0, 0, 0),
                    )
                    screen.blit(text, (dragged_area[0], dragged_area[1] - 30))

                pygame.draw.rect(screen, grid[x][y], (x * 10, y * 10, 10, 10))
                pygame.draw.rect(screen, (44, 44, 44), (x * 10, y * 10, 12, 12), 1)

    non_white_pixels = np.transpose(
        np.where(np.any(grid != (255, 255, 255), axis=-1))
    ).tolist()
    visited = set()
    color_islands = []

    def dfs(x, y, color, grid, visited, color_island, surrounded_by_diff_color):
        stack = [(x, y)]
        visited.add((x, y))

        while stack:
            x, y = stack.pop()
            color_island.append((x, y))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (new_x, new_y) in visited:
                    continue
                if (
                    new_x >= 0
                    and new_x < grid.shape[0]
                    and new_y >= 0
                    and new_y < grid.shape[1]
                ):
                    if tuple(grid[new_x, new_y]) == color:
                        visited.add((new_x, new_y))
                        stack.append((new_x, new_y))
                    else:
                        surrounded_by_diff_color[0] = True

    for x, y in non_white_pixels:
        if (x, y) not in visited:
            color = tuple(grid[x][y])
            color_island = []
            surrounded_by_diff_color = [False]
            dfs(x, y, color, grid, visited, color_island, surrounded_by_diff_color)
            if surrounded_by_diff_color[0]:
                color_islands.append(color_island)

    for color_island in color_islands:
        if surrounded_by_diff_color[0]:
            x_coords, y_coords = np.transpose(color_island)
            center_x, center_y = np.mean(x_coords), np.mean(y_coords)
            color_name = get_color_name(grid[int(center_x)][int(center_y)].tolist())

            text = font.render(color_name, True, (0, 0, 0))
            coord = (int(center_x) * 10, int(center_y) * 10)
            screen.blit(text, coord)

    pygame.display.update()
    pygame.display.flip()
pygame.quit()

screen_width = 1000
screen_height = 1000


def start_screen():
    screen.fill((255, 255, 255))

    title_text = font.render("Color Island", True, (0, 0, 0))
    title_rect = title_text.get_rect()
    title_rect.center = (screen_width // 2, screen_height // 2 - 100)
    screen.blit(title_text, title_rect)

    play_text = font.render("Play", True, (0, 0, 0))
    play_rect = play_text.get_rect()
    play_rect.center = (screen_width // 2, screen_height // 2)
    screen.blit(play_text, play_rect)

    quit_text = font.render("Quit", True, (0, 0, 0))
    quit_rect = quit_text.get_rect()
    quit_rect.center = (screen_width // 2, screen_height // 2 + 50)
    screen.blit(quit_text, quit_rect)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if play_rect.collidepoint(pos):
                    # TODO: add main game loop
                    pass
                elif quit_rect.collidepoint(pos):
                    pygame.quit()
                    sys.exit()


if __name__ == "__main__":
    start_screen()
