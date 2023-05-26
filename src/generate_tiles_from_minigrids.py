## In this file I attempt to generate and save environment visualizations and their textual descriptions.
## then convert those textual descriptions to properly defined annotations + identify the proberties of each of them.
import copy

import gymnasium as gym
import pandas as pd
import numpy as np
import glob
import json
import os
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image, ImageOps
import tensorflow


def extract_context_minigrid(shortname, current_level, current_img_padded, save_dir, tile_dictionary):
    # for traversing through the text file rows
    x = 0
    # for traversing through the image rows
    img_x = 0
    current_level = current_level.split("\n")
    jmax = len(current_level)
    imax = len(current_level[0])
    # outer loop for the row
    for x in range(0, imax, 2):
        # for traversing through the text file columns
        y = 0
        # image_columns
        img_y = 0
        print("Row:", current_level[y])
        for y in range(jmax):
            # candidate tile character
            current_symbol = current_level[y][x:x + 2]
            print("Current Symbol: ", current_symbol)
            # extracting neighbourhood context of candidate tile
            north = '  '
            south = '  '
            west = '  '
            east = '  '
            north_west = '  '
            north_east = '  '
            south_west = '  '
            south_east = '  '

            ##row 1 of data

            if x + 2 < imax and y > 0:
                north_west = current_level[y - 1][x + 2:x + 4]
            if y > 0:
                north = current_level[y - 1][x: x + 2]
            if x > 0 and y > 0:
                north_east = current_level[y - 1][x - 2: x]

            row_1 = str(north_east+ north + north_west)

            # row 2 of data
            if x + 2 < imax:
                west = current_level[y][x + 2:x + 4]
            if x > 0:
                east = current_level[y][x - 2: x]
            row_2 = str(east + current_symbol + west)

            ##row 3 of data
            if x + 2 < imax and  y + 1 < jmax:
                south_west = current_level[y + 1][x + 2: x + 4]
            if y + 1 < jmax:
                south = current_level[y + 1][x: x + 2]
            if x > 0 and y + 1 < jmax:
                south_east = current_level[y + 1][x - 2: x]

            row_3 = str(south_east + south + south_west)

            # identifier string for the context tile
            #sprite_string = str(row_3 + row_2 + row_1)
            sprite_string = str(row_1 + row_2 + row_3)
            print("sprite_string", sprite_string)
            # extract the image
            tile_sprite = img_to_array(current_img_padded)[img_y:img_y + 48, img_x:img_x + 48, :]

            tile_s = img_to_array(current_img_padded)[img_y + 16:img_y + 32, img_x+16:img_x + 32, :]

            print(tile_sprite.shape)
            assert tile_sprite.shape == (48, 48, 3)

            sprite_dir_path = save_dir +"context_data/"+shortname+"/"+ str(current_symbol) + "/"
            symbol_dir = save_dir +"sprites/"+shortname+"/"
            if not os.path.exists(save_dir + "context_data/"+shortname):
                os.mkdir(save_dir + "context_data/"+shortname)
            if not os.path.exists(sprite_dir_path):
                os.mkdir(save_dir + "context_data/"+shortname+"/" + str(current_symbol) + "/")
            if not os.path.exists(symbol_dir):
                os.mkdir(save_dir +"sprites/"+shortname+"/")
            if tile_dictionary.get(current_symbol) is None:
                tile_dictionary[current_symbol] = []
                save_img(symbol_dir + current_symbol + ".png", tile_s, scale=False)

            if sprite_string not in tile_dictionary.get(current_symbol):
                tile_dictionary[current_symbol].append(sprite_string)
                save_img(sprite_dir_path + sprite_string + ".png", tile_sprite, scale=False)

            img_y += 16
        img_x += 16

    return tile_dictionary


def visualize_env(env):

    return env.render()


def describe_env(env):
    s = str(env)
    if "OrderEnforcing" in s:
        sl = s.splitlines()
        env_str = sl[0][-env.width*2:] + "\n"
        for l in sl[1:-1]:
            env_str += l[:env.width*2] + "\n"
        env_str += sl[-1][:env.width*2]
    return env_str


def describe_objects(env):
    objects_desc = dict()
    # ['type','symbol', 'color', 'overlap', 'pickup', 'contain', 'see_behind']
    # Map objects and colors
    COLORS = {
        'R': 'red',
        'G': 'green or grey',
        'B': 'blue',
        'P': 'purple',
        'Y': 'yellow'
    }

    # Map of object types to short string
    OBJECT_TO_STR = {
        'wall': 'W',
        'floor': 'F',
        'door': 'D',
        'key': 'K',
        'ball': 'A',
        'box': 'B',
        'goal': 'G',
        'lava': 'V',
    }

    # Map agent's direction to short string
    AGENT_DIR_TO_STR = {
        0: '>',
        1: 'V',
        2: '<',
        3: '^'
    }

    for j in range(env.grid.height):

        for i in range(env.grid.width):
            if i == env.agent_pos[0] and j == env.agent_pos[1]:
                objects_desc[2*AGENT_DIR_TO_STR[env.agent_dir]] = ['agent']
                continue

            c = env.grid.get(i, j)
            if c == None:
                continue
            obj_type = c.type
            color = c.color
            description = [obj_type]
            overlap = c.can_overlap()
            if overlap:
                description.append("can_overlap")
            pickup = c.can_pickup()
            if pickup:
                description.append("can_pickup")
            contain = c.can_contain()
            if contain:
                description.append("can_contain")
            see_behind = c.see_behind()
            if see_behind:
                description.append("can_see_behind")
            if c.type == 'door':
                if c.is_open:
                    code = '__'
                    description.append("is_open")
                elif c.is_locked:
                    code = 'L' + c.color[0].upper()
                    description.append("is_locked")
                else:
                    code = 'D' + c.color[0].upper()
            else:
                code = OBJECT_TO_STR[c.type] + c.color[0].upper()
            # Not fixed size/features only appear when they are called ['type','symbol', 'color', 'overlap', 'pickup', 'contain', 'see_behind']
            objects_desc[code] = description
    return objects_desc


def make_env(env_key, seed=None):
    env = gym.make(env_key, render_mode="rgb_array", tile_size=16)
    #env.seed(seed)
    return env


def generate_tiles_for_env(shortname, env_dir, env_name, env_seed = 0):

    env = make_env(env_name)
    env.reset()
    env_dict = dict()
    vis_dict = dict()
    obj_desc = dict()
    # Stringify env
    count_duplicate = 0
    while True:
        obj_desc.update(describe_objects(env))
        e_k = describe_env(env)
        if e_k in env_dict:
            count_duplicate += 1
        env_dict[e_k] = copy.deepcopy(env)
        img = array_to_img(visualize_env(env_dict[e_k]))
        img_with_border = ImageOps.expand(img, border=16, fill="black")
        vis_dict[e_k] = img_with_border
        env.reset()
        if count_duplicate >= 2000:
            break

    tile_dictionary = dict()
    i =0
    envs_dir = env_dir+"/"+shortname+"/"
    if not os.path.exists(envs_dir):
        os.mkdir(envs_dir)
    for e_k in env_dict:
        # current_level, current_img_padded, save_dir, tile_dictionary
        save_img(envs_dir+"env_"+str(i)+".png", vis_dict[e_k])
        i += 1
        tile_dictionary = extract_context_minigrid(shortname, e_k, vis_dict[e_k], env_dir+'/', tile_dictionary)

    #save_objects(vis_dict, env_dict, .., ..)
    # Serializing json
    with open("../data/json_files_trimmed_features/"+shortname+".json", "w") as outfile:
        json.dump({'tiles': obj_desc}, outfile)
    with open("../data/context_data/"+shortname+"/candidate_tile_context_"+shortname+".json", "w") as outfile:
        json.dump(tile_dictionary, outfile)


if __name__ == "__main__":
    # Initialize save paths
    env_dir = "../data"
    # Load env
    env_seed = 0

    env_desc = [("MiniGrid-DoorKey-5x5-v0", "doorkey5"),
                ("MiniGrid-Fetch-5x5-N2-v0", "fetch5"),
                ("MiniGrid-GoToDoor-5x5-v0", "godoor5"),
                ("MiniGrid-Unlock-v0", "unlock"),
                ("MiniGrid-LavaGapS5-v0", "lavagap5")]

    for e in env_desc:
        print("Generating for", e[1])
        generate_tiles_for_env(e[1], env_dir, e[0], env_seed)

    print("Done :)   ")