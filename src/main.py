import sys

import pygame
from collections import deque, defaultdict
from pydantic import BaseModel, Field
import random



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
grid = [[(255, 255, 255) for x in range(MapConfig.grid_size * MapConfig.scale)] for y in range(MapConfig.grid_size * MapConfig.scale)]
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
        if (i > color_size*2 and i < color_size*3) and (j > color_size*2 and j < color_size*3):
            grid[i][j] = top_left_color
        elif (i > color_size*6 and i < color_size*7) and (j > color_size*2 and j < color_size*3):
            grid[i][j] = top_right_color
        elif (i > color_size*2 and i < color_size*3) and (j > color_size*6 and j < color_size*7):
            grid[i][j] = bottom_left_color
        elif (i > color_size*6 and i < color_size*7) and (j > color_size*6 and j < color_size*7):
            grid[i][j] = bottom_right_color



running = True
dragging = False
start_pos = None
end_pos = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        def set_color(grid, start_x, start_y):
            color = grid[start_x // 10][start_y // 10]
            return color
        if event.type == pygame.MOUSEBUTTONDOWN:

            start_color = set_color(grid, event.pos[0], event.pos[1])
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

            start_grid_x = start_x // 10
            start_grid_y = start_y // 10
            end_grid_x = end_x // 10
            end_grid_y = end_y // 10

            selected_area = (end_grid_x - start_grid_x + 1) * (end_grid_y - start_grid_y + 1)

            total_color_square = defaultdict(int)
            for x in range(0, 50):
                for y in range(0, 50):
                    total_color_square[tuple(grid[x][y])] += 1

            if selected_area < total_color_square[tuple(start_color)]:
                probability = random.random()
                if probability <= 0.99:
                    for x in range(start_grid_x, end_grid_x + 1):
                        for y in range(start_grid_y, end_grid_y + 1):
                            if 0 <= x < 100 and 0 <= y < 100:
                                grid[x][y] = start_color
                                total_color_square[tuple(start_color)] += 1
                else:
                    inner_color_count = 0
                    for x in range(0, 50):
                        for y in range(0, 50):
                            if grid[x][y] == start_color:
                                inner_color_count += 1
                                if inner_color_count <= selected_area:
                                    grid[x][y] = (255, 255, 255)
                                    total_color_square[tuple(start_color)] -= 1
                                if inner_color_count > selected_area:
                                    break
                        if inner_color_count > selected_area:
                            break
            else:

                print("Selected area cannot be greater than total color square.")
                text = font.render("Selected area cannot be greater than total color square.", True, (0, 0, 0))
                screen.blit(text, (0, 0))




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
                    text = font.render(f"{int(dragged_area[2] / 7)} x {int(dragged_area[3] / 7)}", True, (0, 0, 0))
                    screen.blit(text, (dragged_area[0], dragged_area[1] - 30))

                pygame.draw.rect(screen, grid[x][y], (x * 10, y * 10, 10, 10))
                pygame.draw.rect(screen, (44, 44, 44), (x *10, y *10, 12, 12), 1)




    def flood_fill(grid, start_coord):

        queue = deque([start_coord])
        visited = set()
        color = grid[start_coord[0]][start_coord[1]]

        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))


            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and grid[new_x][new_y] == color:
                    queue.append((new_x, new_y))

        return visited


    color_islands = []
    visited = set()

    for x in range(50):
        for y in range(50):
            if (x, y) not in visited:
                color = grid[x][y]
                if color != (255, 255, 255):
                    island = flood_fill(grid, (x, y))
                    color_islands.append(island)
                    visited.update(island)


    def get_center(color_island):
        x_coords = [x for x, y in color_island]
        y_coords = [y for x, y in color_island]
        center_x = int(sum(x_coords) / len(x_coords))
        center_y = int(sum(y_coords) / len(y_coords))
        return center_x, center_y


    color_islands_dict = {}
    rendered_islands = set()
    for color_island in color_islands:
        center_x, center_y = get_center(color_island)
        color_name = get_color_name(grid[center_x][center_y])

        color_island = tuple(color_island)
        if color_island not in rendered_islands:
            if color_island in color_islands_dict:
                text, coord = color_islands_dict[color_island]
                text = font.render(color_name, True, (0, 0, 0))
                coord = (center_x * 10, center_y * 10)
            else:
                text = font.render(color_name, True, (0, 0, 0))
                coord = (center_x * 10, center_y * 10)
                color_islands_dict[color_island] = (text, coord)

            screen.blit(text, coord)
            rendered_islands.add(color_island)

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
                    #TODO: add main game loop
                    pass
                elif quit_rect.collidepoint(pos):
                    pygame.quit()
                    sys.exit()

if __name__ == "__main__":
    start_screen()

