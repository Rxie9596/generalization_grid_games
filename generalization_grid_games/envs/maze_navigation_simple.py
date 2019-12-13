from .generalization_grid_game import GeneralizationGridGame, create_gym_envs
from .generalization_grid_game import GeneralizationGridGame, create_gym_envs
from .utils import get_asset_path

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon, FancyArrow
import matplotlib.pyplot as plt
import numpy as np


EMPTY = 'empty'
GOAL = 'goal'
AGENT = 'agent'
WALL = 'wall'
LEFT_ARROW = 'left_arrow'
RIGHT_ARROW = 'right_arrow'
UP_ARROW = 'up_arrow'
DOWN_ARROW = 'down_arrow'

ALL_TOKENS = [EMPTY, GOAL, AGENT, WALL, LEFT_ARROW, RIGHT_ARROW, UP_ARROW, DOWN_ARROW]

TOKEN_IMAGES = {
    AGENT : plt.imread(get_asset_path('mice.png')),
    GOAL : plt.imread(get_asset_path('cheese.jpg')),
    WALL : plt.imread(get_asset_path('block.jpg')),
}

HAND_ICON_IMAGE = plt.imread(get_asset_path('hand_icon.png'))

class MazeNavigation_simple(GeneralizationGridGame):

    num_tokens = len(ALL_TOKENS)
    hand_icon = HAND_ICON_IMAGE
    fig_scale = 1.1

    @staticmethod
    def transition(layout, action):
        r, c = action
        token = layout[r, c]
        new_layout = layout.copy()

        if token == UP_ARROW:
            MazeNavigation_simple.step_move_in_direction(new_layout, (-1, 0), AGENT)
        elif token == DOWN_ARROW:
            MazeNavigation_simple.step_move_in_direction(new_layout, (1, 0), AGENT)
        elif token == LEFT_ARROW:
            MazeNavigation_simple.step_move_in_direction(new_layout, (0, -1), AGENT)
        elif token == RIGHT_ARROW:
            MazeNavigation_simple.step_move_in_direction(new_layout, (0, 1), AGENT)

        return new_layout

    @staticmethod
    def compute_reward(state0, action, state1):
        return float(MazeNavigation_simple.compute_done(state1))

    @staticmethod
    def compute_done(state):
        return not np.any(state == GOAL)

    @staticmethod
    def step_move_in_direction(layout, direction, moving_obj_type):
        r, c = np.argwhere(layout == moving_obj_type)[0]
        neighbor_cell = layout[r + direction[0], c + direction[1]]

        if neighbor_cell in [EMPTY, GOAL]:
            next_r, next_c = r + direction[0], c + direction[1]
        else:
            return

        layout[r, c] = EMPTY
        layout[next_r, next_c] = moving_obj_type

    @classmethod
    def draw_token(cls, token, r, c, ax, height, width, token_scale=1.0):
        if token == EMPTY:
            return None

        if 'arrow' in token:
            edge_color = '#888888'
            face_color = 'white'

            drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                     numVertices=4,
                                     radius=0.5 * np.sqrt(2),
                                     orientation=np.pi / 4,
                                     ec=edge_color,
                                     fc=face_color)
            ax.add_patch(drawing)

        if token == LEFT_ARROW:
            arrow_drawing = FancyArrow(c + 0.75, height - 1 - r + 0.5, -0.25,
                                       0.0, width=0.1, fc='purple', head_length=0.2)
            ax.add_patch(arrow_drawing)

        elif token == RIGHT_ARROW:
            arrow_drawing = FancyArrow(c + 0.25, height - 1 - r + 0.5, 0.25,
                                       0.0, width=0.1, fc='purple', head_length=0.2)
            ax.add_patch(arrow_drawing)

        elif token == UP_ARROW:
            arrow_drawing = FancyArrow(c + 0.5, height - 1 - r + 0.25, 0.0,
                                       0.25, width=0.1, fc='purple', head_length=0.2)
            ax.add_patch(arrow_drawing)

        elif token == DOWN_ARROW:
            arrow_drawing = FancyArrow(c + 0.5, height - 1 - r + 0.75, 0.0,
                                       -0.25, width=0.1, fc='purple', head_length=0.2)
            ax.add_patch(arrow_drawing)

        else:
            im = TOKEN_IMAGES[token]
            oi = OffsetImage(im, zoom=cls.fig_scale * (token_scale / max(height, width) ** 0.5))
            box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)

            ax.add_artist(box)

            return box

    @classmethod
    def initialize_figure(cls, height, width):
        fig, ax = GeneralizationGridGame.initialize_figure(height, width)

        # Draw a white grid in the background
        for r in range(height):
            for c in range(width):
                edge_color = '#888888'
                face_color = 'white'

                drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                         numVertices=4,
                                         radius=0.5 * np.sqrt(2),
                                         orientation=np.pi / 4,
                                         ec=edge_color,
                                         fc=face_color)
                ax.add_patch(drawing)

        return fig, ax


E = EMPTY
G = GOAL
A = AGENT
W = WALL
U = UP_ARROW
D = DOWN_ARROW
L = LEFT_ARROW
R = RIGHT_ARROW

layout0 = [
    [U, D, L, R, W, W, W],
    [W, W, W, W, W, W, W],
    [W, E, E, G, E, E, W],
    [W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W],
    [W, A, W, E, W, E, W],
    [W, W, W, W, W, W, W],
]
layout1 = [
    [U, D, L, R, W, W, W],
    [W, W, W, W, W, W, W],
    [W, E, E, G, E, E, W],
    [W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W],
    [W, E, W, A, W, E, W],
    [W, W, W, W, W, W, W],
]
layout2 = [
    [U, D, L, R, W, W, W],
    [W, W, W, W, W, W, W],
    [W, E, E, G, E, E, W],
    [W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W],
    [W, E, W, E, W, A, W],
    [W, W, W, W, W, W, W],
]
layout3 = [
    [U, D, L, R, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, G, E, E, E, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, A, W, E, W, E, W, E, W, E, W],
    [W, W, W, W, W, W, W, W, W, W, W],
]
layout4 = [
    [U, D, L, R, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, G, E, E, E, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, A, W, E, W, E, W, E, W],
    [W, W, W, W, W, W, W, W, W, W, W],
]
layout5 = [
    [U, D, L, R, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, G, E, E, E, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, A, W, E, W, E, W],
    [W, W, W, W, W, W, W, W, W, W, W],
]
layout6 = [
    [U, D, L, R, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, G, E, E, E, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, A, W, E, W],
    [W, W, W, W, W, W, W, W, W, W, W],
]
layout7 = [
    [U, D, L, R, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, G, E, E, E, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, E, W],
    [W, E, W, E, W, E, W, E, W, A, W],
    [W, W, W, W, W, W, W, W, W, W, W],
]

layouts = [layout0, layout1, layout2, layout3, layout4, layout5, layout6,
           layout7]

create_gym_envs(MazeNavigation_simple, layouts, globals())