""" Simple maze environment

"""
import numpy as np
import os
import torch

from nsrl.base_classes import Environment

# matplotlib.use('agg')
# matplotlib.use('qt5agg')
from nsrl.helper.pytorch import device
import plotly.graph_objs as go
import plotly.io as pio

from nsrl.helper.plot import scatter_3d_multi_color
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

class MyEnv(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, **kwargs):

        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0
        self._size_maze=kwargs.get('size_maze', 8)
        self._higher_dim_obs=kwargs.get("higher_dim_obs", False)
        self._maze_walls = kwargs.get('maze_walls', True)
        self.create_map()
        self.intern_dim = kwargs.get('intern_dim', 2)
        self._trajectory = []

        self._init_position = [self._size_maze // 2, self._size_maze // 2]
        if self._size_maze > 20 and self._maze_walls:
            self._init_position = [1, 1]

    def create_map(self):
        self._map=np.zeros((self._size_maze,self._size_maze))
        self._map[-1,:]=1
        self._map[0,:]=1
        self._map[:,0]=1
        self._map[:,-1]=1
        if self._maze_walls:
            if self._size_maze < 20:
                self._map[:,self._size_maze//2]=1
                self._map[self._size_maze // 2, self._size_maze // 2] = 0
                self._pos_agent = [self._size_maze // 2, self._size_maze // 2]
                self._pos_goal = [self._size_maze - 2, self._size_maze - 2]
            else:
                self._map[:, self._size_maze//2] = 1
                self._map[self._size_maze//2, :] = 1

                self._pos_agent = [1, 1]
                self._pos_goal = [self._size_maze - 2, self._size_maze - 2]

                # Make openings in walls
                self._map[self._size_maze//4, self._size_maze//2] = 0
                self._map[self._size_maze//4 - 1, self._size_maze//2] = 0
                self._map[self._size_maze//4 - 2, self._size_maze//2] = 0

                self._map[self._size_maze//2, self._size_maze//4] = 0
                self._map[self._size_maze//2, self._size_maze//4 - 1] = 0
                self._map[self._size_maze//2, self._size_maze//4 - 2] = 0

                self._map[self._size_maze//2, self._size_maze//2 + self._size_maze//4] = 0
                self._map[self._size_maze//2, self._size_maze//2 + self._size_maze//4 + 1] = 0
                self._map[self._size_maze//2, self._size_maze//2 + self._size_maze//4 + 2] = 0

                self._map[self._size_maze//2 + self._size_maze//4, self._size_maze//2] = 0
                self._map[self._size_maze//2 + self._size_maze//4 + 1, self._size_maze//2] = 0
                self._map[self._size_maze//2 + self._size_maze//4 + 2, self._size_maze//2] = 0

                print("here")
        else:
            self._pos_agent = [self._size_maze // 2, self._size_maze // 2]
            self._pos_goal=[self._size_maze-2,self._size_maze-2]

                
    def reset(self, mode):
        self.create_map()

        self._map[self._size_maze//2,self._size_maze//2]=0
        
        if mode == MyEnv.VALIDATION_MODE:
            if self._mode != MyEnv.VALIDATION_MODE:
                self._mode = MyEnv.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0
                
            else:
                self._mode_episode_count += 1
        elif self._mode != -1:
            self._mode = -1
        
        # Setting the starting position of the agent
        self._pos_agent = copy.deepcopy(self._init_position)

        # clear trajectory
        self._trajectory = [copy.deepcopy(self._init_position)]


        return [1 * [self._size_maze * [self._size_maze * [0]]]]

    def loadState(self, dataset, previous_env=None):
        # There's two ways of doing this: replay actions and using last state. Here we use last state.
        if previous_env is not None:
            self.__dict__ = previous_env.__dict__.copy()
            return

        # we also need to fill the trajectories
        for data in dataset.observations()[0]:
            y, x = np.where(data == 0.5)
            self._trajectory.append([y.item(), x.item()])

        last_obs = dataset.observations()[0][-1]

        # Now we make sure our hyperparams are correct
        self._size_maze = last_obs.shape[0]
        self._maze_walls = (last_obs[1:-1, 1:-1] == 1).any()
        self.create_map()
        y, x = np.where(last_obs == 0.5)
        self._pos_agent = [y.item(), x.item()]


    def act(self, action):
        """Applies the agent action [action] on the environment.

        Parameters
        -----------
        action : int
            The action selected by the agent to operate on the environment. Should be an identifier 
            included between 0 included and nActions() excluded.
        """

        self._cur_action=action
        if(action==0): # UP
            if(self._map[self._pos_agent[0]-1,self._pos_agent[1]]==0):
                self._pos_agent[0]=self._pos_agent[0]-1
        elif(action==1): # DOWN
            if(self._map[self._pos_agent[0]+1,self._pos_agent[1]]==0):
                self._pos_agent[0]=self._pos_agent[0]+1
        elif(action==2): # LEFT
            if(self._map[self._pos_agent[0],self._pos_agent[1]-1]==0):
                self._pos_agent[1]=self._pos_agent[1]-1
        elif(action==3): # RIGHT
            if(self._map[self._pos_agent[0],self._pos_agent[1]+1]==0):
                self._pos_agent[1]=self._pos_agent[1]+1

        copy_pos_agent = copy.deepcopy(self._pos_agent)
        self._trajectory.append(copy_pos_agent)

        # There is no reward in this simple environment
        self.reward = 0

        self._mode_score += self.reward
        return self.reward

    def getAllPossibleStates(self):
        all_possib_inp=[] # Will store all possible inputs (=observation) for the agent

        for y_a in range(self._size_maze):
            for x_a in range(self._size_maze):
                state=copy.deepcopy(self._map)
                state[self._size_maze//2,self._size_maze//2]=0
                if(state[x_a,y_a]==0):
                    if(self._higher_dim_obs==True):
                        all_possib_inp.append(self.get_higher_dim_obs([[x_a,y_a]],[self._pos_goal]))
                    else:
                        state[x_a,y_a]=0.5
                        all_possib_inp.append(state)
        return np.array(all_possib_inp)

    def summarizePerformance(self, test_data_set, learning_algo, *args, **kwargs):
        """ Plot of the low-dimensional representation of the environment built by the model
        """
        save_image = kwargs.get('save_image', False)
        action_meanings = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        plot_all_possible_inputs = kwargs.get('plot_all_possible_inputs', False)
        if self.intern_dim is None:
            raise Exception("Cannot summarize performance if no internal representation is learnt")

        all_possib_inp=[] # Will store all possible inputs (=observation) for the agent

        all_positions = []

        for y_a in range(self._size_maze):
            for x_a in range(self._size_maze):
                state=copy.deepcopy(self._map)
                state[self._size_maze//2,self._size_maze//2]=0
                if(state[x_a,y_a]==0 and (plot_all_possible_inputs or [y_a, x_a] in self._trajectory)):
                    all_positions.append([x_a, y_a])
                    if(self._higher_dim_obs==True):
                        all_possib_inp.append(self.get_higher_dim_obs([[x_a,y_a]],[self._pos_goal]))
                    else:
                        state[y_a, x_a]=0.5
                        all_possib_inp.append(state)
        
        all_possib_inp=np.expand_dims(np.array(all_possib_inp,dtype='float'),axis=1)
        
        all_possib_inp = torch.from_numpy(all_possib_inp).float().to(device)
        with torch.no_grad():
            # all_possib_abs_states=learning_algo.encoder.predict(all_possib_inp)

            observation_set = np.unique(test_data_set.observations()[0], axis=0)
            position_set = []
            for obs in observation_set:
                y, x = np.where(obs == 0.5)
                position_set.append([x.item(), y.item()])

            n = all_possib_inp.shape[0]

            historics = torch.from_numpy(observation_set).float().to(device)
            torch_abs_states=learning_algo.encoder.predict(historics)
            # if(abs_states.ndim==4):
            #     abs_states=np.transpose(abs_states, (0, 3, 1, 2))    # data_format='channels_last' --> 'channels_first'

            # WARNING - this does not work with observation set
            # actions=test_data_set.actions()[0:n]

            if self.inTerminalState() == False:
                self._mode_episode_count += 1
            print("== Mean score per episode is {} over {} episodes ==".format(self._mode_score / (self._mode_episode_count+0.0001), self._mode_episode_count))


            m = cm.ScalarMappable(cmap=cm.jet)

            abs_states = torch_abs_states.cpu().detach().numpy()
            # all_possib_abs_states = all_possib_abs_states.cpu().detach().numpy()

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
        if self.intern_dim == 2:
            opacities = [0.15, 0.4, 0.65, 0.9]
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

            colors = ['blue', 'orange', 'green']
            middle = self._size_maze//2

            for i in range(3):
            # for color in colors:
                # position = all_travelled_positions[length_block[i][0]:length_block[i][1]]
                # all_travelled_positions
                indices = [idx for idx, (pos_x, pos_y) in enumerate(position_set) if pos_x > middle]
                if i == 0:
                    # Left side of map

                    # indices where we're at left side
                    indices = [idx for idx, (pos_x, pos_y) in enumerate(position_set) if pos_x < middle]

                elif i == 1:
                    indices = [idx for idx, (pos_x, pos_y) in enumerate(position_set) if pos_x == middle]

                if(self.intern_dim==2):
                    # x = all_possib_abs_states[length_block[i][0]:length_block[i][1],0]
                    # y = all_possib_abs_states[length_block[i][0]:length_block[i][1],1]
                    # right side of map

                    x = abs_states[indices, 0]
                    y = abs_states[indices, 1]

                    scatter = go.Scatter(x=x, y=y,
                                         mode='markers+text',
                                         marker=dict(symbol='x', size=10, color=colors[i]),
                                         text=[str(position_set[i]) for i in indices],
                                         textposition='top center')
                    trace_data.append(scatter)

        else:
            trace = scatter_3d_multi_color([{'x': x, 'y': y, 'z': z, 'color': 'blue'}])
            trace_data.append(trace)

        fig = dict(data=trace_data)

        if save_image:
            pio.write_image(fig, 'pytorch/fig_base_'+str(learning_algo.repr_update_counter)+'.png')

        return fig

        # fig = plt.figure()
        # if(self.intern_dim==2):
        #     ax = fig.add_subplot(111)
        #     ax.set_xlabel(r'$X_1$')
        #     ax.set_ylabel(r'$X_2$')
        # else:
        #     ax = fig.add_subplot(111,projection='3d')
        #     ax.set_xlabel(r'$X_1$')
        #     ax.set_ylabel(r'$X_2$')
        #     ax.set_zlabel(r'$X_3$')
        #
        # # Plot the estimated transitions
        # for i in range(n):
        #     # pdb.set_trace()
        #     predicted1=learning_algo.transition.predict(torch.cat((torch.from_numpy(all_possib_abs_states[i:i+1]).float() ,torch.from_numpy(np.array([[1,0,0,0]])).float()),-1).to(device)).cpu().detach().numpy()
        #     predicted2=learning_algo.transition.predict(torch.cat((torch.from_numpy(all_possib_abs_states[i:i+1]).float() ,torch.from_numpy(np.array([[0,1,0,0]])).float()),-1).to(device)).cpu().detach().numpy()
        #     predicted3=learning_algo.transition.predict(torch.cat((torch.from_numpy(all_possib_abs_states[i:i+1]).float() ,torch.from_numpy(np.array([[0,0,1,0]])).float()),-1).to(device)).cpu().detach().numpy()
        #     predicted4=learning_algo.transition.predict(torch.cat((torch.from_numpy(all_possib_abs_states[i:i+1]).float() ,torch.from_numpy(np.array([[0,0,0,1]])).float()),-1).to(device)).cpu().detach().numpy()
        #     # predicted1=learning_algo.transition.predict([all_possib_abs_states[i:i+1],np.array([[1,0,0,0]])])
        #     # predicted2=learning_algo.transition.predict([all_possib_abs_states[i:i+1],np.array([[0,1,0,0]])])
        #     # predicted3=learning_algo.transition.predict([all_possib_abs_states[i:i+1],np.array([[0,0,1,0]])])
        #     # predicted4=learning_algo.transition.predict([all_possib_abs_states[i:i+1],np.array([[0,0,0,1]])])
        #     if(self.intern_dim==2):
        #         ax.plot(np.concatenate([x[i:i+1],predicted1[0,:1]]), np.concatenate([y[i:i+1],predicted1[0,1:2]]), color="0.9", alpha=0.75)
        #         ax.plot(np.concatenate([x[i:i+1],predicted2[0,:1]]), np.concatenate([y[i:i+1],predicted2[0,1:2]]), color="0.65", alpha=0.75)
        #         ax.plot(np.concatenate([x[i:i+1],predicted3[0,:1]]), np.concatenate([y[i:i+1],predicted3[0,1:2]]), color="0.4", alpha=0.75)
        #         ax.plot(np.concatenate([x[i:i+1],predicted4[0,:1]]), np.concatenate([y[i:i+1],predicted4[0,1:2]]), color="0.15", alpha=0.75)
        #     else:
        #         ax.plot(np.concatenate([x[i:i+1],predicted1[0,:1]]), np.concatenate([y[i:i+1],predicted1[0,1:2]]), np.concatenate([z[i:i+1],predicted1[0,2:3]]), color="0.9", alpha=0.75)
        #         ax.plot(np.concatenate([x[i:i+1],predicted2[0,:1]]), np.concatenate([y[i:i+1],predicted2[0,1:2]]), np.concatenate([z[i:i+1],predicted2[0,2:3]]), color="0.65", alpha=0.75)
        #         ax.plot(np.concatenate([x[i:i+1],predicted3[0,:1]]), np.concatenate([y[i:i+1],predicted3[0,1:2]]), np.concatenate([z[i:i+1],predicted3[0,2:3]]), color="0.4", alpha=0.75)
        #         ax.plot(np.concatenate([x[i:i+1],predicted4[0,:1]]), np.concatenate([y[i:i+1],predicted4[0,1:2]]), np.concatenate([z[i:i+1],predicted4[0,2:3]]), color="0.15", alpha=0.75)
        #
        # # Plot the dots at each time step depending on the action taken
        # # length_block=[[0,18],[18,19],[19,31]]
        # colors = ['blue', 'orange', 'green']
        # middle = self._size_maze//2
        #
        # for i in range(3):
        # # for color in colors:
        #     # position = all_travelled_positions[length_block[i][0]:length_block[i][1]]
        #     position = all_positions
        #     # all_travelled_positions
        #     indices = [idx for idx, (pos_x, pos_y) in enumerate(position) if pos_x > middle]
        #     if i == 0:
        #         # Left side of map
        #
        #         # indices where we're at left side
        #         indices = [idx for idx, (pos_x, pos_y) in enumerate(position) if pos_x < middle]
        #
        #     elif i == 1:
        #         indices = [idx for idx, (pos_x, pos_y) in enumerate(position) if pos_x == middle]
        #
        #     if(self.intern_dim==2):
        #         # x = all_possib_abs_states[length_block[i][0]:length_block[i][1],0]
        #         # y = all_possib_abs_states[length_block[i][0]:length_block[i][1],1]
        #         # right side of map
        #
        #         x = abs_states[indices, 0]
        #         y = abs_states[indices, 1]
        #
        #         line3 = ax.scatter(x, y, c=colors[i], marker='x', edgecolors='k', alpha=0.5, s=100)
        #
        #         # Annotate the points
        #         for j, k in enumerate(indices):
        #             color = 'black'
        #             # if position[k] in all_travelled_positions:
        #             #     color = 'red'
        #             ax.text(x[j], y[j], str(position[k]), color=color)
        #     else:
        #         # Might have to have an option to make a plotly plot here instead for visdom.
        #         x = abs_states[indices, 0]
        #         y = abs_states[indices, 1]
        #         z = abs_states[indices, 2]
        #         line3 = ax.scatter(x, y, z, marker='x', depthshade=True, edgecolors='k', alpha=0.5, s=50)
        #         # Annotate the points
        #         for j, k in enumerate(indices):
        #             ax.text(x[j], y[j], z[j], str(position[k]))
        # if(self.intern_dim==2):
        #     axes_lims=[ax.get_xlim(),ax.get_ylim()]
        # else:
        #     axes_lims=[ax.get_xlim(),ax.get_ylim(),ax.get_zlim()]
        #
        # # Plot the legend for transition estimates
        # box1b = TextArea(" Estimated transitions (action 0, 1, 2 and 3): ", textprops=dict(color="k"))
        # box2b = DrawingArea(90, 20, 0, 0)
        # el1b = Rectangle((5, 10), 15,2, fc="0.9", alpha=0.75)
        # el2b = Rectangle((25, 10), 15,2, fc="0.65", alpha=0.75)
        # el3b = Rectangle((45, 10), 15,2, fc="0.4", alpha=0.75)
        # el4b = Rectangle((65, 10), 15,2, fc="0.15", alpha=0.75)
        # box2b.add_artist(el1b)
        # box2b.add_artist(el2b)
        # box2b.add_artist(el3b)
        # box2b.add_artist(el4b)
        #
        # boxb = HPacker(children=[box1b, box2b],
        #               align="center",
        #               pad=0, sep=5)
        #
        # anchored_box = AnchoredOffsetbox(loc=3,
        #                                  child=boxb, pad=0.,
        #                                  frameon=True,
        #                                  bbox_to_anchor=(0., 0.98),
        #                                  bbox_transform=ax.transAxes,
        #                                  borderpad=0.,
        #                                  )
        # ax.add_artist(anchored_box)
        #
        #
        # #plt.show()
        # plt.savefig('pytorch/fig_base_'+str(learning_algo.repr_update_counter)+'.pdf')


        # return fig, plt

    def inputDimensions(self):
        if(self._higher_dim_obs==True):
            return [(1,self._size_maze*6,self._size_maze*6)]
        else:
            return [(1,self._size_maze,self._size_maze)]

    def observationType(self, subject):
        return np.float

    def nActions(self):
        return 4

    def observe(self, higher_dim=False):
        obs=copy.deepcopy(self._map)
                
        obs[self._pos_agent[0],self._pos_agent[1]]=0.5                
        if(self._higher_dim_obs or higher_dim):
            obs=self.get_higher_dim_obs([self._pos_agent],[self._pos_goal])
            
        return [obs]

    def map_trajectory(self, save_image=False, img_name='default'):
        obs = self.get_higher_dim_obs([self._pos_agent], [self._pos_goal])

        # Add points for all places
        for x, y in self._trajectory:
            pos_x = x * 6
            pos_y = y * 6
            obs[pos_x + 3, pos_y + 3] = min(obs[pos_x + 3, pos_y + 3] + 0.1, 1.0)

        for i in range(len(self._trajectory) - 1):
            if np.sum(np.array(self._trajectory[i]) - np.array(self._trajectory[i + 1])) > 1:
                continue
            bgn = self._trajectory[i]
            bgn_x = bgn[0] * 6 + 3
            bgn_y = bgn[1] * 6 + 3

            end = self._trajectory[i + 1]
            end_x = end[0] * 6 + 3
            end_y = end[1] * 6 + 3

            color = (obs[bgn_x, bgn_y] + obs[end_x, end_y]) / 2 + 0.1
            if abs(bgn_x - end_x) > 0:
                x1, x2 = sorted([bgn_x, end_x])
                obs[x1:x2, bgn_y] = color
            elif abs(bgn_y - end_y) > 0:
                y1, y2 = sorted([bgn_y, end_y])
                obs[bgn_x, y1:y2] = color



        if save_image:
            img_file_name = os.path.join('trajectories', img_name + '.png')
            cwd = os.getcwd()
            img_full_file_name = os.path.join(cwd, img_file_name)
            if os.path.exists(img_full_file_name):
                os.remove(img_file_name)
            plt.imsave(img_file_name, obs, cmap='gray_r')
        # plt.imshow(obs, cmap='gray_r')
        # plt.show()

        return obs

    def get_higher_dim_obs(self,indices_agent=None):
        """ Obtain the high-dimensional observation from indices of the agent position and the indices of the reward positions.
        """
        if indices_agent is None:
            indices_agent = [self._init_position]
        obs=copy.deepcopy(self._map)
        obs=obs/2.2
        obs=np.repeat(np.repeat(obs, 6, axis=0),6, axis=1)
        # agent repr
        agent_obs = np.ones((6,6))
        # agent_obs=np.zeros((6,6))
        #
        # agent_obs[0,2]=0.7
        # agent_obs[1,0:5]=0.8
        # agent_obs[2,1:4]=0.8
        # agent_obs[3,1:4]=0.8
        # agent_obs[4,1]=0.8
        # agent_obs[4,3]=0.8
        # agent_obs[5,0:2]=0.8
        # agent_obs[5,3:5]=0.8
        
        # reward repr
        reward_obs=np.zeros((6,6))
        #reward_obs[:,1]=0.8
        #reward_obs[0,1:4]=0.7
        #reward_obs[1,3]=0.8
        #reward_obs[2,1:4]=0.7
        #reward_obs[4,2]=0.8
        #reward_obs[5,2:4]=0.8
        
        # for i in indices_reward:
        #     obs[i[0]*6:(i[0]+1)*6:,i[1]*6:(i[1]+1)*6]=reward_obs

        for i in indices_agent:
            obs[i[0]*6:(i[0]+1)*6:,i[1]*6:(i[1]+1)*6]=agent_obs
            
        plt.imshow(obs, cmap='gray_r')

        plt.show()
        return obs


    def inTerminalState(self):
        # Uncomment the following lines to add some cases where the episode terminates.
        # This is used to show how the environment representation interpret cases where 
        # part of the environment could not be explored.
#        if((self._pos_agent[0]<=1 and self._cur_action==0) ):
#            return True
        return False

        # If there is a goal, then terminates the environment when the goas is reached.
        #if (self._pos_agent==self._pos_goal):
        #    return True
        #else:
        #    return False



if __name__ == "__main__":
    env = MyEnv(0, higher_dim_obs=True)
    env.map_trajectory()

