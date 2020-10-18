import curses
import sys
import copy
import torch
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

from nsrl.helper.pytorch import device
from nsrl.base_classes import Environment

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import rendering
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

MAZE_ART = [
    # First we have our 10x10 maze
    ['##########',
     '#P     # #',
     '######   #',
     '#    # ###',
     '###    # #',
     '# #### # #',
     '# #      #',
     '#   ######',
     '# #      #',
     '##########'],

    # Now the 15x15 multi-step maze
    # ['###############',
    #  '#P  ####### D #',
    #  '#   #######   #',
    #  '#   TTTTTTT   #',
    #  '#   TTTTTTT   #',
    #  '#   TTTTTTT   #',
    #  '#   #######   #',
    #  '#   #######   #',
    #  '#   #######   #',
    #  '#   TTTTTTT   #',
    #  '#   TTTTTTT   #',
    #  '#   TTTTTTT   #',
    #  '#   #######   #',
    #  '# K #######   #',
    #  '###############',
    #  ],

    ['###############',
     '#P  TTTTTTT   #',
     '#   TTTTTTT   #',
     '#   TTTTTTT   #',
     '#   #######   #',
     '#   #######   #',
     '#   #######   #',
     '#   #######   #',
     '#   #######   #',
     '#   #######   #',
     '#   #######   #',
     '#   #######   #',
     '#   #######   #',
     '# K ####### D #',
     '###############',
     ],
    # Now the 21x21 maze
    ['#######################',
     '#P            #     # #',
     '# # # # ####### ##### #',
     '# # # # # # # #     # #',
     '# # ### # # # # ##### #',
     '# # # #               #',
     '# # # ####### ### # # #',
     '# #         #   # # # #',
     '# # # ### # # # # # # #',
     '# # #   # # # # # # # #',
     '# # # ### # # ### ### #',
     '# # #   # # #   # #   #',
     '# ### # # ### # ### # #',
     '#   # # #   # #   # # #',
     '# ### ##### # # #######',
     '# #       # # #       #',
     '# ### ####### ### ### #',
     '# #         #   #   # #',
     '##### # # ####### #####',
     '#     # #       #     #',
     '##### # ### ### # ### #',
     '#     #   #   # #   # #',
     '#######################']
]

class PlayerSprite(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character):
        super(PlayerSprite, self).__init__(corner, position, character, impassable='#T')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers

        if actions == 0:  # go upward?
            self._north(board, the_plot)
        elif actions == 1:  # go downward?
            self._south(board, the_plot)
        elif actions == 2:  # go leftward?
            self._west(board, the_plot)
        elif actions == 3:  # go rightward?
            self._east(board, the_plot)

class KeyDrape(plab_things.Drape):
    """A `Drape` for the key.
    This `Drape` simply disappears if the player sprite steps on any element where
    its curtain is True, setting the 'has_key' flag in the Plot as it goes.
    I guess we'll give the player a reward for key collection, too.
    """

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if self.curtain[things['P'].position]:
          the_plot['has_key'] = True
          the_plot.add_reward(1.0)
        if the_plot.get('has_key'): self.curtain[:] = False  # Only one key.

class DoneDrape(plab_things.Drape):
    """
    'Drape' for the exit
    Checks to see if 'player' has key. If he/she/they do, then done
    """

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if self.curtain[things['P'].position] and the_plot.get('has_key'):
            the_plot.add_reward(1.0)
            the_plot.terminate_episode()

class TempWallsDrape(plab_things.Drape):
    """
    'Drape' for temporary walls that disappear after taking key
    """

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if the_plot.get('has_key'): self.curtain[:] = False

MAZE_FG_COLORS = {
    ' ': (870, 838, 678), # Maze floor
    '#': (428, 135, 0),   # Maze walls
    'P': (388, 400, 999), # Player
    'K': (178, 178, 178), # Key
    'T': (570, 538, 378), # Wall
    'D': (770, 400, 400)  # Reward
}


def make_game(level):
    maze_art = MAZE_ART[level]
    what_lies_beneath = ' '
    sprites = {'P': PlayerSprite}
    drapes = {'K': KeyDrape, 'D': DoneDrape, 'T': TempWallsDrape}

    update_schedule = [['D', 'P', 'K', 'T']]
    return ascii_art.ascii_art_to_game(
        maze_art, what_lies_beneath, sprites, drapes,
        update_schedule=update_schedule)

class MazeEnv(Environment):
    def __init__(self, rng=None, size_maze=10, intern_dim=2, **kwargs):
        """
        Wrapper for OpenMaze environment. Currently rng not implemented for mazes.
        :param rng:
        :param kwargs:
        """
        self._mode = -1
        self._size_maze = size_maze
        assert self._size_maze == 10 or self._size_maze == 21 or self._size_maze == 15

        self._input_dims = [(1, self._size_maze, self._size_maze)]

        self.intern_dim = intern_dim
        self._trajectory = [[1, 1]] # maybe there's a programmatic way of doing this?

        self._game = None
        self._observation = None
        self._reward = None
        self._discount = None
        self._map = None
        self._agent_repr = None
        self._keys = {}


    def get_maze_index(self):
        mapping = {10: 0, 15: 1, 21: 3}
        return mapping[self._size_maze]

    def reset(self, mode=-1):
        self._game = make_game(self.get_maze_index())
        self._observation, self._reward, self._discount = self._game.its_showtime()
        if self._size_maze == 15: # we may want to add more items here later for multiple keys
            y, x = np.nonzero(self._observation.layers['K'])
            self._keys['K'] = self._observation.board[y, x].item()
        self._map = self._observation.layers['#'].astype(int)
        y, x = np.nonzero(self._observation.layers['P'])
        self._agent_repr = self._observation.board[y, x].item()
        self._trajectory = [[1, 1]] # maybe there's a programmatic way of doing this?

    def loadState(self, dataset):
        raise NotImplementedError

    def act(self, action):
        self._observation, self._reward, self._discount = self._game.play(action)
        pos_y, pos_x = np.where(self._observation.layers['P'])
        self._trajectory.append([pos_y.item(), pos_x.item()])

        if self._reward is None:
            return 0
        return self._reward

    @property
    def trajectory(self):
        return self._trajectory

    def inputDimensions(self):
        return self._input_dims

    def observationType(self, subject):
        return np.float

    def nActions(self):
        return 4

    def observe(self):
        assert self._observation is not None
        obs = copy.deepcopy(self._observation.board)

        return [obs]

    def inTerminalState(self):
        return self._game.game_over

    def summarizePerformance(self, test_data_set, learning_algo, *args, **kwargs):
        save_image = kwargs.get('save_image', False)
        action_meanings = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        plot_all_possible_inputs = kwargs.get('plot_all_possible_inputs', False)
        if self.intern_dim is None:
            raise Exception("Cannot summarize performance if no internal representation is learnt")

        with torch.no_grad():
            observation_set = np.unique(test_data_set.observations()[0], axis=0)

            position_set = []
            state_set = []
            for obs in observation_set:
                y, x = np.where(obs == self._agent_repr)
                position_set.append([x.item(), y.item()])
                if self._size_maze == 15:
                    state = [x.item(), y.item()]
                    for key, value in self._keys.items():
                        to_append = 0
                        if value not in obs:
                            to_append = 1
                        state.append(to_append)
                    state_set.append(state)

            historics = torch.from_numpy(observation_set).float().to(device)
            torch_abs_states=learning_algo.encoder.predict(historics)

            abs_states = torch_abs_states.cpu().detach().numpy()


            x = np.array(abs_states)[:,0]
            y = np.array(abs_states)[:,1]
            if(self.intern_dim>2):
                z = np.array(abs_states)[:,2]

            trans_by_action_idx = []
            stacked_transitions = np.eye(self.nActions())
            # each element of this list should be a transition

            for one_hot_action in stacked_transitions:
                repeated_one_hot_actions = torch.from_numpy(np.repeat(one_hot_action[None, :], abs_states.shape[0], axis=0)).float().to(device)
                res = torch.cat([torch_abs_states, repeated_one_hot_actions], dim=-1)
                transitions = learning_algo.transition(res).detach().cpu().numpy()

                trans_by_action_idx.append(transitions)

        trace_data = []
        unit = 256 // len(x)
        opacity_unit = 1 / self.nActions()
        opacities = [(i + 1) * opacity_unit for i in range(self.nActions())]
        colors = None
        if self._size_maze == 15 and state_set:
            colors = ['blue' if k == 1 else 'orange' for _, _, k in state_set]
        else:
            colors = [f"rgb({int(i * unit)}, {int(unit * (len(x) - i))}, 0)" for i in range(len(x))]
        if self.intern_dim == 2:
            for trans, aname, opacity in zip(trans_by_action_idx, action_meanings, opacities):
                plot_x = []
                plot_y = []
                for x_o, y_o, x_y_n in zip(x, y, trans):
                    plot_x += [x_o, x_y_n[0], None]
                    plot_y += [y_o, x_y_n[1], None]
                trace_data.append(
                    go.Scatter(x=plot_x,
                               y=plot_y,
                               line=dict(color='rgba(0, 0, 0, ' + str(opacity) + ')'),
                               marker=dict(size=1),
                               name=aname))
            scatter = go.Scatter(x=x, y=y, mode='markers+text',
                                 text=state_set if state_set else position_set,
                                 textposition='top center',
                                 marker=dict(symbol='x', size=10,
                                             color=colors))
            trace_data.append(scatter)

        elif self.intern_dim == 3:
            for trans, aname, opacity in zip(trans_by_action_idx, action_meanings, opacities):
                plot_x = []
                plot_y = []
                plot_z = []
                for x_o, y_o, z_o, x_y_z_n in zip(x, y, z, trans):
                    plot_x += [x_o, x_y_z_n[0], None]
                    plot_y += [y_o, x_y_z_n[1], None]
                    plot_z += [z_o, x_y_z_n[2], None]

                trace_data.append(
                    go.Scatter3d(
                        x=plot_x, y=plot_y, z=plot_z,
                        line=dict(color='rgba(0, 0, 0, ' + str(opacity) + ')'),
                        marker=dict(size=1),
                        name=aname))
            scatter = go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers+text',
                                   text=state_set if state_set else position_set,
                                   textposition='top center',
                                   marker=dict(symbol='circle',
                                               size=3,
                                               color=colors))
            trace_data.append(scatter)
        fig = dict(data=trace_data)

        if save_image:
            pio.write_image(fig, 'pytorch/fig_base_'+str(learning_algo.repr_update_counter)+'.png')

        return fig

def main(argv=()):
    game = make_game(int(argv[1]) if len(argv) > 1 else 0)

    # Make a CursesUi to play it with.
    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                         -1: 4,
                         'q': 5, 'Q': 5},
        delay=100,
        colour_fg=MAZE_FG_COLORS)
    ui.play(game)

if __name__ == "__main__":
    main(sys.argv)