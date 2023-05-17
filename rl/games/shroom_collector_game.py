import pygame
import sys
import os
from pygame.locals import *
import numpy as np
import random
import time


class Constants:
    TILES_X, TILES_Y = 20, 20
    WIDTH, HEIGHT = 800, 800
    WINDOW_SIZE = (WIDTH, HEIGHT)
    TILE_SIZE = (TILES_X, TILES_Y)
    SPRITE_SIZE = 40
    COLLECTOR_COLOR = (255, 255, 255, 255)
    GRASS_COLOR = (0, 255, 0, 255)
    LAVA_COLOR = (255, 0, 0, 255)
    SAND_COLOR = (255, 255, 0, 255)
    WATER_COLOR = (0, 0, 255, 255)
    MUSHROOM_COLOR = (0, 255, 255, 255)
    NOTHING_COLOR = (0, 0, 0, 255)
    TILE_ONE_HOT = {
        GRASS_COLOR: 1,
        LAVA_COLOR: 2,
        SAND_COLOR: 3,
        WATER_COLOR: 4,
        COLLECTOR_COLOR: 5,
        MUSHROOM_COLOR: 6,
        NOTHING_COLOR: 0

    }
    # inverted look up version
    TILE_ONE_HOT_LOOKUP = new_dict = {v: k for k, v in TILE_ONE_HOT.items()}

    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTIONS = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT]

    REASON_LAVA = 0
    REASON_ENERGY = 1
    REASON_SHROOMS_COLLECTED = 2
    SHROOM_GOAL = 5
    COLLECTOR_VISION_DISTANCE = 20
    # Used for calculating the area around the collector to grab
    COLLECTOR_VISION_BOX = COLLECTOR_VISION_DISTANCE * 2 + 1


class ShroomCollectorGame:
    def __init__(self, ui_enabled: bool = False,
                 random_shroom_count: int = 0,
                 positional_shrooms=[],
                 respawn_mushrooms=True,
                 sand_to_lava=False,
                 map_file_name="grassy_island_map.png"):
        # Initialize Pygame
        self.ui_enabled: bool = ui_enabled
        self.loaded_assets = False
        self.map_file_name = map_file_name
        self.sand_to_lava = sand_to_lava
        if self.ui_enabled:
            self.WINDOW = pygame.display.set_mode(Constants.WINDOW_SIZE)
            pygame.display.set_caption("Shroom Collector")

        self.terrain_layer = np.zeros(Constants.TILE_SIZE)
        self.mushroom_layer = np.zeros(Constants.TILE_SIZE)
        self.sfx_layer = np.zeros(Constants.TILE_SIZE)

        # Load the map image
        script_path = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_path, '../../assets', self.map_file_name)
        self.map_image: pygame.image = pygame.image.load(image_path)
        # Character position for bottom left corner

        self.character_position: [int] = [0, 0]
        self.prev_character_position: [int] = self.character_position
        self.energy: int = 100
        self.done: bool = False
        self.game_over_reason: int = -1
        self.shrooms_collected: int = 0
        self.found_shroom: bool = False
        self.respawn_mushrooms: bool = respawn_mushrooms
        self.positional_shrooms = positional_shrooms
        self.random_shrooms_count = random_shroom_count

        # Load sprite images
        if self.ui_enabled and not self.loaded_assets:
            self.image_sprites = {
                Constants.LAVA_COLOR: pygame.image.load(os.path.join(script_path, '../../assets', 'lava_sprite2.png')),
                Constants.GRASS_COLOR: pygame.image.load(os.path.join(script_path, '../../assets', 'grass_sprite3.png')),
                Constants.COLLECTOR_COLOR: pygame.image.load(os.path.join(script_path, '../../assets', 'grass_sprite3.png')),
                Constants.SAND_COLOR: pygame.image.load(os.path.join(script_path, '../../assets', 'sand_sprite2.png')),
                Constants.WATER_COLOR: pygame.image.load(os.path.join(script_path, '../../assets', 'water_sprite2.png')),
                Constants.MUSHROOM_COLOR: pygame.image.load(os.path.join(script_path, '../../assets', 'shroom_sprite2.png')),
            }
            self.character_sprite = pygame.image.load(os.path.join(script_path, '../../assets', 'shroom_collector.png'))
            self.loaded_assets = True

        self.reset()

    def _get_shroom_collector_start(self):
        for y in range(Constants.TILES_Y):
            for x in range(Constants.TILES_X):
                tile_color = Constants.TILE_ONE_HOT_LOOKUP[self.terrain_layer[x][y]]
                if tile_color == Constants.COLLECTOR_COLOR:
                    return [x, y]
        print(f"Unable to find shroom collector pixel")
        sys.exit(1)



    def reset(self) -> None:
        self.terrain_layer = np.zeros(Constants.TILE_SIZE)
        self.mushroom_layer = np.zeros(Constants.TILE_SIZE)
        self.sfx_layer = np.zeros(Constants.TILE_SIZE)

        # Load the map image
        script_path = os.path.dirname(os.path.abspath(__file__))

        script_path = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_path, '../../assets', self.map_file_name)
        self.map_image: pygame.image = pygame.image.load(image_path)

        self.map_image = pygame.image.load(image_path)
        # Character position for bottom left corner
        self.energy: int = 100
        self.done: bool = False
        self.game_over_reason: int = -1
        self.shrooms_collected: int = 0
        self.found_shroom = False

        self._load_layers()
        self.character_position = self._get_shroom_collector_start()
        self.prev_character_position = self.character_position


        for i in range(self.random_shrooms_count):
            self._place_random_shroom()

        if self.positional_shrooms is not None:
            for shroom in self.positional_shrooms:
                self._place_a_shroom(x=shroom[0], y=shroom[1])

    def _load_layers(self) -> None:
        for y in range(Constants.TILES_Y):
            for x in range(Constants.TILES_X):
                pixel_color = self.map_image.get_at((x, y))
                pixel_color = tuple(pixel_color)
                if pixel_color in Constants.TILE_ONE_HOT:
                    self.terrain_layer[x][y] = Constants.TILE_ONE_HOT[pixel_color]
                else:
                    self.terrain_layer[x][y] = Constants.TILE_ONE_HOT[Constants.NOTHING_COLOR]

    def _place_a_shroom(self, x: int, y: int):
        self.mushroom_layer[x][y] = Constants.TILE_ONE_HOT[Constants.MUSHROOM_COLOR]
        return False

    def _place_random_shroom(self):
        mushroom_spots = []
        for y in range(Constants.TILES_Y):
            for x in range(Constants.TILES_X):
                tile_color = Constants.TILE_ONE_HOT_LOOKUP[self.terrain_layer[x][y]]
                mushroom_spot_color = Constants.TILE_ONE_HOT_LOOKUP[self.mushroom_layer[x][y]]

                if tile_color == Constants.GRASS_COLOR and mushroom_spot_color == Constants.NOTHING_COLOR:
                    mushroom_spots.append((x, y))

        if len(mushroom_spots) > 0:
            random_spot = random.choice(mushroom_spots)
            self._place_a_shroom(x=random_spot[0], y=random_spot[1])
            # self.mushroom_layer[random_spot[0]][random_spot[1]] = Constants.TILE_ONE_HOT[Constants.MUSHROOM_COLOR]

    def draw_terrain(self):
        for y in range(Constants.TILES_Y):
            for x in range(Constants.TILES_X):
                tile_color = Constants.TILE_ONE_HOT_LOOKUP[self.terrain_layer[x][y]]
                # pixel_color = self.map_image.get_at((x, y))
                tile_color = tuple(tile_color)
                if tile_color in self.image_sprites:
                    sprite = self.image_sprites[tile_color]
                    if sprite is not None:
                        self.WINDOW.blit(sprite, (x * Constants.SPRITE_SIZE, y * Constants.SPRITE_SIZE))

    def draw_character(self):
        self.WINDOW.blit(self.character_sprite, (self.character_position[0] * Constants.SPRITE_SIZE, self.character_position[1] * Constants.SPRITE_SIZE))


    def draw_shrooms(self):
        for y in range(Constants.TILES_Y):
            for x in range(Constants.TILES_X):
                tile_color = Constants.TILE_ONE_HOT_LOOKUP[self.mushroom_layer[x][y]]
                tile_color = tuple(tile_color)
                if tile_color in self.image_sprites:
                    sprite = self.image_sprites[tile_color]
                    if sprite is not None:
                        self.WINDOW.blit(sprite, (x * Constants.SPRITE_SIZE, y * Constants.SPRITE_SIZE))

    def _is_blocker(self, x, y):
        return self.terrain_layer[x][y] in [Constants.TILE_ONE_HOT[Constants.WATER_COLOR], Constants.TILE_ONE_HOT[Constants.NOTHING_COLOR]]

    def get_available_action(self):
        available_actions = []

        x = self.character_position[0]
        y = self.character_position[1]

        if x - 1 >= 0 and not self._is_blocker(x - 1, y):
            available_actions.append(Constants.ACTION_LEFT)
        if x + 1 < Constants.TILES_X and not self._is_blocker(x + 1, y):
            available_actions.append(Constants.ACTION_RIGHT)

        if y - 1 >= 0 and not self._is_blocker(x, y - 1):
            available_actions.append(Constants.ACTION_UP)

        if y + 1 < Constants.TILES_Y and not self._is_blocker(x, y+1):
            available_actions.append(Constants.ACTION_DOWN)

        return available_actions

    def get_surface_under_character(self):
        # if self.character_position != self.prev_character_position:
        x = self.character_position[0]
        y = self.character_position[1]
        return self.terrain_layer[x][y]
        # else:
        #     return 0

    def _update_character_position(self, curr, prev):
        x_curr = curr[0]
        y_curr = curr[1]

        x_prev = prev[0]
        y_prev = prev[1]

        mushroom_potential = self.mushroom_layer[x_curr][y_curr]
        if mushroom_potential == Constants.TILE_ONE_HOT[Constants.MUSHROOM_COLOR]:
            self.shrooms_collected += 1
            self.energy += 50
            self.mushroom_layer[x_curr][y_curr] = 0
            if self.respawn_mushrooms:
                self._place_random_shroom()
            self.found_shroom = True

        self.mushroom_layer[x_prev][y_prev] = Constants.TILE_ONE_HOT[Constants.COLLECTOR_COLOR]
        self.mushroom_layer[x_prev][y_prev] = Constants.TILE_ONE_HOT[Constants.NOTHING_COLOR]

        if self.sand_to_lava:
            if self.terrain_layer[x_prev][y_prev] == Constants.TILE_ONE_HOT[Constants.SAND_COLOR]:
                self.terrain_layer[x_prev][y_prev] = Constants.TILE_ONE_HOT[Constants.LAVA_COLOR]

        # self.mushroom_layer[prev[0], prev[1]] = 0
        # self.mushroom_layer[curr[0], curr[1]] = ShroomCollectorConstants.
    def _apply_action(self, action: int):
        self.prev_character_position = self.character_position.copy()
        self.energy -= 1
        if action == Constants.ACTION_UP:
            self.character_position[1] -= 1
        elif action == Constants.ACTION_RIGHT:
            self.character_position[0] += 1
        elif action == Constants.ACTION_DOWN:
            self.character_position[1] += 1
        elif action == Constants.ACTION_LEFT:
            self.character_position[0] -= 1
        self._update_character_position(self.character_position, self.prev_character_position)

    def check_game_over_state(self):
        current_square = self.get_surface_under_character()
        if current_square == Constants.TILE_ONE_HOT[Constants.LAVA_COLOR]:
            self.done = True
            self.game_over_reason = Constants.REASON_LAVA

        if self.energy <= 0:
            self.done = True
            self.game_over_reason = Constants.REASON_ENERGY

        if self.shrooms_collected >= Constants.SHROOM_GOAL:
            self.done = True
            self.game_over_reason = Constants.REASON_SHROOMS_COLLECTED


        return self.done, self.game_over_reason



    def step(self, action: int = None):
        if action is not None and not self.done:
            self._apply_action(action)

        if self.ui_enabled:
            self.render()

    def render(self):
        self.draw_terrain()
        self.draw_shrooms()
        self.draw_character()
        pygame.display.set_caption(
            f"Shroom Collector --- {self.shrooms_collected} / 5 Shrooms --- Energy: {self.energy}")


def main():
    clock = pygame.time.Clock()
    lg = ShroomCollectorGame(ui_enabled=True)
    action = None
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            # Handle character movement
            if event.type == KEYDOWN:
                if event.key == K_w:
                    # lg.character_position[1] -= 1
                    action = Constants.ACTION_UP
                elif event.key == K_s:
                    # lg.character_position[1] += 1
                    action = Constants.ACTION_DOWN
                elif event.key == K_a:
                    # lg.character_position[0] -= 1
                    action = Constants.ACTION_LEFT
                elif event.key == K_d:
                    # lg.character_position[0] += 1
                    action = Constants.ACTION_RIGHT

        # Draw the map and character
        if not action in lg.get_available_action():
            action = None

        lg.step(action)
        action = None

        pygame.display.update()
        over, reason = lg.check_game_over_state()
        clock.tick(10)

        if over:
            lg.reset()

if __name__ == "__main__":
    main()
