import random
import sys
import time
from threading import Thread

import numpy as np
import pygame
import scipy
import torch
import torch.nn as nn
from pydantic import BaseModel, Field


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


grid_size = 100
player_size = MapConfig.player_size

if grid_size % player_size != 0:
    grid_size += player_size - (grid_size % player_size)

grid = np.full(
    (grid_size, grid_size, 3),
    (255, 255, 255),
)
screen_size = grid_size * player_size
screen = pygame.display.set_mode((MapConfig.window_size + 200, MapConfig.window_size))
pygame.display.set_caption(MapConfig.caption)

font = pygame.font.SysFont(MapConfig.font_type, MapConfig.font_size)

grid_size = 100
color_size = int(grid_size * 0.06)
player_size = 5


grid[:player_size, :player_size] = MapConfig.top_left_color
grid[:player_size, -player_size:] = MapConfig.top_right_color
grid[-player_size:, :player_size] = MapConfig.bottom_left_color
grid[-player_size:, -player_size:] = MapConfig.bottom_right_color

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


COLORS = [
    np.array([255, 0, 0]),
    np.array([0, 255, 0]),
    np.array([0, 150, 255]),
    np.array([255, 215, 0]),
]
# Define the size of the color count text box
color_count_box_width = 200
color_count_box_height = screen_size // 2

# Define the starting position of the color count text box
color_count_box_x = MapConfig.window_size
color_count_box_y = screen_size // 8


pygame.draw.rect(
    screen, (255, 255, 255), (color_count_box_x, 0, color_count_box_width, screen_size)
)

# Print the number of squares for each color, one below the other
for i, color in enumerate(COLORS):
    num_squares = np.count_nonzero(np.all(grid == color, axis=2))
    color_name = get_color_name(color.tolist())
    color_text = f"{color_name}: {num_squares}"
    color_text_render = font.render(color_text, True, (0, 0, 0))
    color_text_rect = color_text_render.get_rect()
    color_text_rect.center = (
        color_count_box_x + color_count_box_width // 2,
        color_count_box_y + i * color_count_box_height // 2,
    )
    screen.blit(color_text_render, color_text_rect)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def get_state(grid, color):
    color_mask = (grid == color).all(axis=2)
    color_total_area = np.sum(color_mask)
    if color_total_area == 0:
        return np.zeros((1, 5))  # Return a zero vector with shape (1, 5)
    color_mean = np.mean(grid[color_mask], axis=0) / 255
    opponent_colors = [c for c in COLORS if not np.array_equal(c, color)]
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
    ).reshape(1, 5)


def ai_function(color, grid, model, actions, eps):
    opponent_colors = [c for c in COLORS if not np.array_equal(c, color)]
    opponent_colors = [
        c
        for c in opponent_colors
        if np.any((grid == c).all(axis=2) & (grid == color).all(axis=2))
    ]

    color_mask = (grid == color).all(axis=2)
    color_total_area = np.sum(color_mask)
    # calculate distance from center of map
    map_center = np.array([grid.shape[0] // 2, grid.shape[1] // 2])
    color_center = np.mean(np.argwhere(color_mask), axis=0)
    color_distance = np.linalg.norm(color_center - map_center)

    # adjust eps based on distance from center
    eps_adjustment = 1.0 + 0.1 * (color_distance / max(map_center))
    eps *= eps_adjustment
    if color_total_area == 0:
        print(f"No {color} zone found.")
        return 0, 0, 0, True
    max_area = min(color_total_area // 2 + 10, np.count_nonzero(color_mask))

    target_opponent_color = None
    if color_total_area > (grid.shape[0] * grid.shape[1]) / 2:
        # Find the smallest opponent color nearby
        min_opponent_area = float("inf")
        target_opponent_color = None
        for opponent_color in opponent_colors:
            opponent_mask = (grid == opponent_color).all(axis=2)
            opponent_area = np.sum(opponent_mask)
            if opponent_area < min_opponent_area:
                min_opponent_area = opponent_area
                target_opponent_color = opponent_color

        if target_opponent_color is not None:
            # Find the edge of the smallest opponent color
            opponent_mask = (grid == target_opponent_color).all(axis=2)
            edge_mask = (
                scipy.ndimage.morphology.binary_dilation(opponent_mask) & ~opponent_mask
            )
            edge_points = np.argwhere(edge_mask)
            start_pos = edge_points[np.random.randint(len(edge_points))]
            start_x, start_y = start_pos
            print(
                f"{color} zone is too big. Starting to eat the edges of the smallest opponent color."
            )
        else:
            start_pos = np.argwhere(color_mask)[np.random.randint(color_total_area)]
            start_x, start_y = start_pos
    else:
        start_pos = np.argwhere(color_mask)[np.random.randint(color_total_area)]
        start_x, start_y = start_pos

    # Move towards the center of the smallest opponent color
    if target_opponent_color is not None:
        opponent_mask = (grid == target_opponent_color).all(axis=2)
        opponent_center = np.mean(np.argwhere(opponent_mask), axis=0)
        dx, dy = opponent_center - np.array([start_x, start_y])
        if dx > 0:
            end_x = np.clip(start_x + np.random.randint(1, dx + 1), 0, 99)
        else:
            end_x = np.clip(start_x - np.random.randint(1, -dx + 1), 0, 99)
        if dy > 0:
            end_y = np.clip(start_y + np.random.randint(1, dy + 1), 0, 99)
        else:
            end_y = np.clip(start_y - np.random.randint(1, -dy + 1), 0, 99)
    else:
        end_x = np.clip(start_x + np.random.randint(1, max_area + 1), 0, 99)
        end_y = np.clip(start_y + np.random.randint(1, max_area + 1), 0, 99)

    start_x, end_x = np.sort([start_x, end_x])
    start_y, end_y = np.sort([start_y, end_y])
    selected_area = np.abs((end_x - start_x + 1) * (end_y - start_y + 1))
    grid_mask = np.zeros_like(grid)
    grid_mask[start_x : end_x + 1, start_y : end_y + 1] = 1
    white_mask = (grid == (255, 255, 255)).all(axis=2)[..., np.newaxis]
    opponent_mask = np.zeros_like(grid_mask)

    for opponent_color in opponent_colors:
        opponent_mask |= (grid == opponent_color).all(axis=2)[..., np.newaxis]
        if (
            (grid_mask & opponent_mask).any()
            or (grid_mask & (grid != color).all(axis=2)[..., np.newaxis]).any()
            or (grid_mask & white_mask).sum() < selected_area
        ):
            print(
                "Selected area overlaps with opponent color or existing color or white spots."
            )
            continue
    reward = get_reward(grid, color, start_x, start_y, end_x, end_y)
    if selected_area <= color_total_area and reward > 0:
        grid[start_x : end_x + 1, start_y : end_y + 1] = color
        print(
            f"Area bought. selected_area={selected_area}, color_total_area={color_total_area}, reward={reward} ✅"
        )
        return color_total_area, selected_area, reward
    else:
        print(
            f"Area not bought. selected_area={selected_area}, color_total_area={color_total_area}, reward={reward} ❌"
        )
        return color_total_area, selected_area, reward


def select_action(model, state, actions, eps):
    state = torch.Tensor(state).unsqueeze(0)
    q_values = model(state).detach().numpy()
    if random.random() < eps:
        action = random.choice(actions)
    else:
        action = actions[np.argmax(q_values)]
    return action


def get_reward(grid, color, start_x, start_y, end_x, end_y):
    selected_area = (end_x - start_x + 1) * (end_y - start_y + 1)
    total_color_square = np.sum((grid == color).all(axis=2))
    white_mask = (grid == (255, 255, 255)).all(axis=2)

    if total_color_square == 10000:
        reward = 1.0
        return reward
    if selected_area > total_color_square:
        reward = -1.0
        return reward
    else:
        reward = 0.2
        has_white = (white_mask[start_x : end_x + 1, start_y : end_y + 1]).any()
        has_color = (
            (grid[start_x : end_x + 1, start_y : end_y + 1] == color).all(axis=2)
        ).any()

        if (
            total_color_square > 0
            and selected_area >= total_color_square * 0.5
            and has_color
        ):
            reward += 0.7

        if (
            total_color_square > 0
            and selected_area >= total_color_square * 0.75
            and has_color
        ):
            reward += 0.8

        if has_white and has_color:
            reward += 0.4
            return reward

        if has_color:
            reward -= 0.3
            return reward

        for c in COLORS:
            if not np.array_equal(c, color):
                if (
                    (grid[start_x : end_x + 1, start_y : end_y + 1] == c).all(axis=2)
                ).any() and has_color:
                    reward += 0.8
                    return reward

            if not np.array_equal(c, color):
                if (
                    (
                        (grid[start_x : end_x + 1, start_y : end_y + 1] == c).all(
                            axis=2
                        )
                    ).any()
                    and has_color
                    and has_white
                ):
                    reward += 0.7
                    return reward

        if selected_area == 0:
            reward -= 0.2
            return reward

        if (grid[start_x : end_x + 1, start_y : end_y + 1] == color).all(axis=2).all():
            reward -= 0.2
            return reward

        if (grid[start_x : end_x + 1, start_y : end_y + 1] != color).all(
            axis=2
        ).any() or (grid[start_x : end_x + 1, start_y : end_y + 1] == color).all(
            axis=2
        ).all():
            reward -= 0.3

    return reward


def train_model(model, grid, color, actions, episodes, optimizer, loss_fn):
    color = np.array(color)
    for episode in range(episodes):
        print("Episode: ", episode, "color: ", color)
        eps = max(0.1, 0.95 - 0.05 * episode)
        state = get_state(grid, color)
        action = select_action(model, state, actions, 0.9999)
        if action is not None:
            try:
                start_x, start_y = action[0] * 10, action[1] * 10
                end_x, end_y = action[2] * 10 + 9, action[3] * 10 + 9
                _, selected_area, reward = ai_function(color, grid, model, actions, eps)
                next_state = get_state(grid, color)
                target = reward + 0.99 * np.max(
                    model(torch.Tensor(next_state).unsqueeze(0)).detach().numpy()
                )
                target_tensor = torch.Tensor([[target]])
                loss = loss_fn(model(torch.Tensor(state).unsqueeze(0)), target_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except ValueError:
                pass
            if np.sum((grid == color).all(axis=2)) == 0:
                print(f"{color} has lost the game!")
                break
    time.sleep(0.001)


def train_models_threaded(models, grid, colors, actions, episodes, optimizers, loss_fn):
    threads = []
    for i, (color, model, optimizer) in enumerate(zip(colors, models, optimizers)):
        thread = Thread(
            target=train_model,
            args=(model, grid, color, actions, episodes, optimizer, loss_fn),
        )
        thread.start()
        threads.append(thread)
        time.sleep(0.1)

    for thread in threads:
        thread.join()
        time.sleep(0.1)

    for i, model in enumerate(models):
        torch.save(model.state_dict(), f"model_{i}.pth")


models = [torch.nn.Sequential() for _ in range(4)]
for model in models:
    model.add_module("fc1", torch.nn.Linear(5, 24))
    model.add_module("relu1", torch.nn.ReLU())
    model.add_module("fc2", torch.nn.Linear(24, 48))
    model.add_module("relu2", torch.nn.ReLU())
    model.add_module("fc3", torch.nn.Linear(48, 3))
optimizer1 = torch.optim.Adam(models[0].parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(models[1].parameters(), lr=0.001)
optimizer3 = torch.optim.Adam(models[2].parameters(), lr=0.001)
optimizer4 = torch.optim.Adam(models[3].parameters(), lr=0.001)
optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
loss_fn = torch.nn.MSELoss()

# train_models_threaded(
#    models,
#    grid,
#    COLORS,
#    actions,
#    episodes=10000,
#    optimizers=optimizers,
#    loss_fn=loss_fn,
# )


def get_local_models(num_models):
    models = []
    for i in range(num_models):
        model_path = f"model_{i}.pth"
        model = torch.nn.Sequential()
        model.add_module("fc1", torch.nn.Linear(5, 24))
        model.add_module("relu1", torch.nn.ReLU())
        model.add_module("fc2", torch.nn.Linear(24, 48))
        model.add_module("relu2", torch.nn.ReLU())
        model.add_module("fc3", torch.nn.Linear(48, 3))
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models.append(model)
    return models


while running:
    initial_counts = [np.sum((grid == color).all(axis=2)) for color in COLORS]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        def set_color(grid, pos):
            if pos[0] > 1000 or pos[1] > 1000:
                return None
            color = grid[pos[0] // 10, pos[1] // 10]
            return color

        if event.type == pygame.MOUSEBUTTONDOWN:
            start_color = set_color(grid, event.pos)
            start_pos = event.pos
            dragging = True

        if event.type == pygame.MOUSEBUTTONUP:
            # Update scoreboard on the left
            for color, model in zip(COLORS, models):
                state = get_state(grid, color)
                action = select_action(model, state, actions, 0.999)
                start_x, start_y = action[0] * 10, action[1] * 10
                end_x, end_y = action[2] * 10 + 9, action[3] * 10 + 9
                ai_function(color, grid, model, actions, 0.999)
            pygame.draw.rect(
                screen,
                (255, 255, 255),
                (color_count_box_x, 0, color_count_box_width, screen_size),
            )
            for i, color in enumerate(COLORS):
                num_squares = np.count_nonzero(np.all(grid == color, axis=2))
                color_name = get_color_name(color.tolist())
                color_text = f"{color_name}: {num_squares}"
                color_text_render = font.render(color_text, True, (0, 0, 0))
                color_text_rect = color_text_render.get_rect()
                color_text_rect.center = (
                    color_count_box_x + color_count_box_width // 2,
                    color_count_box_y + i * (color_count_box_height // 2),
                )
                pygame.draw.line(
                    screen,
                    (200, 200, 200),
                    (color_count_box_x - 200, color_text_rect.centery + 3 * 10),
                    (
                        color_count_box_x + color_count_box_width,
                        color_text_rect.centery + 3 * 10,
                    ),
                    1,
                )
                if num_squares - initial_counts[i] > 0:
                    color_text_render = font.render(color_text, True, (0, 255, 0))
                elif num_squares - initial_counts[i] < 0:
                    color_text_render = font.render(color_text, True, (255, 0, 0))
                screen.blit(color_text_render, color_text_rect)
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
