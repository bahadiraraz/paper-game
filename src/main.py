import sys

import pygame
from collections import deque, defaultdict
from pydantic import BaseModel, Field
import random



class MapConfig(BaseModel):
    top_left_color: list = Field([255, 0, 0])
    top_right_color: list = Field([0, 255, 0])
    bottom_left_color: list = Field([0, 0, 255])
    bottom_right_color: list = Field([255, 255, 0])
    window_size: int = Field(1000, ge=1, le=1000)
    grid_size: int = Field(100, ge=1, le=100)
    caption: str = Field("map")
    font_size: int = Field(30, ge=1, le=100)
    font_type: str = Field("Arial")

pygame.init()
MapConfig = MapConfig()

screen = pygame.display.set_mode((MapConfig.window_size, MapConfig.window_size))
pygame.display.set_caption(MapConfig.caption)
top_left_color = MapConfig.top_left_color
top_right_color = MapConfig.top_right_color
bottom_left_color = MapConfig.bottom_left_color
bottom_right_color = MapConfig.bottom_right_color
grid = [[(255, 255, 255) for x in range(100)] for y in range(100)]
font = pygame.font.SysFont(MapConfig.font_type, MapConfig.font_size)



def get_color_name(color):
    if color == [255, 0, 0]:
        return "Red"
    elif color == [0, 255, 0]:
        return "Green"
    elif color == [0, 0, 255]:
        return "Blue"
    elif color == [255, 255, 0]:
        return "Yellow"


for x in range(5):
    for y in range(5):
        grid[x][y] = top_left_color
for x in range(95, 100):
    for y in range(5):
        grid[x][y] = top_right_color
for x in range(5):
    for y in range(95, 100):
        grid[x][y] = bottom_left_color
for x in range(95, 100):
    for y in range(95, 100):
        grid[x][y] = bottom_right_color


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
            for x in range(0, 100):
                for y in range(0, 100):
                    total_color_square[tuple(grid[x][y])] += 1

            if selected_area < total_color_square[tuple(start_color)]:
                probability = random.random()
                if probability <= 0.9:
                    for x in range(start_grid_x, end_grid_x + 1):
                        for y in range(start_grid_y, end_grid_y + 1):
                            if 0 <= x < 100 and 0 <= y < 100:
                                grid[x][y] = start_color
                                total_color_square[tuple(start_color)] += 1
                else:
                    inner_color_count = 0
                    for x in range(0, 100):
                        for y in range(0, 100):
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
                text = font.render("Selected area cannot be greater than total color square.", True, (255, 0, 0))
                screen.blit(text, (500, 500))


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
                    print(dragged_area)
                    pygame.draw.rect(screen, (0, 0, 255), dragged_area, 1)
                    text = font.render(f"{int(dragged_area[2] / 7)} x {int(dragged_area[3] / 7)}", True, (0, 0, 0))
                    screen.blit(text, (dragged_area[0], dragged_area[1] - 30))

                pygame.draw.rect(screen, grid[x][y], (x * 10, y * 10, 10, 10))
                pygame.draw.rect(screen, (0, 0, 0), (x *10, y *10, 12, 12), 1)




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
    for x in range(100):
        for y in range(100):
            if (x, y) not in visited:
                color = grid[x][y]
                if color != (255, 255, 255):
                    island = flood_fill(grid, (x, y))
                    color_islands.append(island)
                    visited.update(island)

    for color_island in color_islands:
        color_count = 0
        center_x = 0
        center_y = 0

        for coord in color_island:
            x, y = coord
            color_count += 1
            center_x += x
            center_y += y

        center_x = center_x // color_count
        center_y = center_y // color_count

        color_name = get_color_name(grid[center_x][center_y])
        text = font.render(color_name, True, (0, 0, 0))
        screen.blit(text, (center_x * 10, center_y * 10))

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

