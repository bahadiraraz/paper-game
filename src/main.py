import random
import sys

import numpy as np
import pygame
from pydantic import BaseModel, Field


class MapConfig(BaseModel):
    top_left_color: list = Field([255, 0, 0])
    top_right_color: list = Field([0, 255, 0])
    bottom_left_color: list = Field([0, 150, 255])
    bottom_right_color: list = Field([255, 215, 0])
    window_size: int = Field(500, ge=1, le=500)
    grid_size: int = Field(100, ge=1, le=100)
    player_size: int = Field(5, ge=1, le=10)
    scale: int = Field(1, ge=1, le=20)
    caption: str = Field("map")
    font_size: int = Field(20, ge=1, le=100)
    font_type: str = Field("arial")


pygame.init()
MapConfig = MapConfig()

screen = pygame.display.set_mode((MapConfig.window_size, MapConfig.window_size))
pygame.display.set_caption(MapConfig.caption)
top_left_color = MapConfig.top_left_color
top_right_color = MapConfig.top_right_color
bottom_left_color = MapConfig.bottom_left_color
bottom_right_color = MapConfig.bottom_right_color
grid = np.full(
    (MapConfig.grid_size * MapConfig.scale, MapConfig.grid_size * MapConfig.scale, 3),
    (255, 255, 255),
)
font = pygame.font.SysFont(MapConfig.font_type, MapConfig.font_size)


def get_color_name(color):
    if color == MapConfig.top_left_color:
        return "Red"
    if color == MapConfig.top_right_color:
        return "Green"
    if color == MapConfig.bottom_left_color:
        return "Blue"
    if color == MapConfig.bottom_right_color:
        return "Yellow"


grid_size = MapConfig.grid_size * MapConfig.scale
color_size = int(grid_size * 0.06)
player_size = MapConfig.player_size * MapConfig.scale

for i in range(grid_size):
    for j in range(grid_size):
        if (i > color_size * 2 and i < color_size * 3) and (
            j > color_size * 2 and j < color_size * 3
        ):
            grid[i][j] = top_left_color
        elif (i > color_size * 6 and i < color_size * 7) and (
            j > color_size * 2 and j < color_size * 3
        ):
            grid[i][j] = top_right_color
        elif (i > color_size * 2 and i < color_size * 3) and (
            j > color_size * 6 and j < color_size * 7
        ):
            grid[i][j] = bottom_left_color
        elif (i > color_size * 6 and i < color_size * 7) and (
            j > color_size * 6 and j < color_size * 7
        ):
            grid[i][j] = bottom_right_color


running = True
dragging = False
start_pos = None
end_pos = None


def bot_function(color):
    color_indices = np.where(np.all(grid == color, axis=-1))
    color_zone = np.column_stack(color_indices)

    if color_zone.size == 0:
        print(f"No {color} zone found.")
        return

    start_pos = color_zone[np.random.randint(color_zone.shape[0])]
    start_x, start_y = start_pos
    end_x = np.clip(start_x + np.random.randint(-10, 11), 0, 49)
    end_y = np.clip(start_y + np.random.randint(-10, 11), 0, 49)
    end_pos = (end_x, end_y)  # noqa F841

    start_x, end_x = np.sort([start_x, end_x])
    start_y, end_y = np.sort([start_y, end_y])

    selected_area = (end_x - start_x + 1) * (end_y - start_y + 1)

    total_color_square = np.count_nonzero(grid == color)

    if selected_area <= total_color_square:
        grid[start_x : end_x + 1, start_y : end_y + 1] = color
    else:
        print(f"Selected area cannot be greater than total {color} square.")


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        for i in [
            top_left_color,
            top_right_color,
            bottom_left_color,
            bottom_right_color,
        ]:
            bot_function(i)

        def set_color(grid, pos):
            color = grid[pos[0] // 10, pos[1] // 10]
            return color

        if event.type == pygame.MOUSEBUTTONDOWN:
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

        for x in range(50):
            for y in range(50):
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
