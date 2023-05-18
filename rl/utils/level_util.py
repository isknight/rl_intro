from rl.games.shroom_collector_game import ShroomCollectorGame


def get_demo(ui_enabled=True):
    positional_shrooms = []
    for i in range(19):
        positional_shrooms.append([i + 1, 10])
    game = ShroomCollectorGame(ui_enabled=ui_enabled,
                               positional_shrooms=positional_shrooms,
                               respawn_mushrooms=False,
                               map_file_name="demo_map.png",
                               shroom_count_goal=19)
    return game


def get_grassy_island(ui_enabled=True):
    game = ShroomCollectorGame(ui_enabled=ui_enabled,
                               random_shroom_count=5,
                               respawn_mushrooms=False,
                               map_file_name="grassy_island_map.png")
    return game


def get_maze(ui_enabled=True):
    game = ShroomCollectorGame(ui_enabled=ui_enabled,
                               random_shroom_count=0,
                               respawn_mushrooms=False,
                               positional_shrooms=[[19, 17]],
                               map_file_name="maze_map.png")
    return game


def get_eruption(ui_enabled=True):
    game = ShroomCollectorGame(ui_enabled=ui_enabled,
                               random_shroom_count=10,
                               respawn_mushrooms=True,
                               sand_to_lava=False,
                               map_file_name="eruption_map.png",
                               shroom_count_goal=10)
    return game
