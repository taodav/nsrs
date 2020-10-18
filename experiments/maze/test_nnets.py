#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from nsrl.helper.pytorch import device
from nsrl.learning_algos.NSRS_pytorch import NSRS
from nsrl.agent import SEAgent
from simple_maze_env_pytorch import MyEnv as simple_maze_env
from nsrl.helper.plot import Plotter
from nsrl.helper.data import Bunch
from definitions import ROOT_DIR
from nsrl.helper.knn import ranked_avg_knn_scores, avg_knn_scores, batch_knn, batch_count_scaled_knn

def step_env(action, agent):
    obs = agent._environment.observe()
    reward = agent._environment.act(action)
    is_terminal = agent._environment.inTerminalState()

    agent._addSample(obs, action, reward, is_terminal)

def plot_obs_with_markings(agent):
    with torch.no_grad():
        for m in agent._learning_algo.all_models: m.eval()
        dataset = agent._dataset
        #         observation_set = np.unique(dataset.observations()[0], axis=0)
        observation_set = dataset.observations()[0]
        trajectory = []
        for obs in observation_set:
            y, x = np.where(obs == 0.5)
            trajectory.append([y.item(), x.item()])
        n = observation_set.shape[0]
        observations = torch.from_numpy(observation_set).float().to(device)
        abs_states=learning_algo.encoder.predict(observations)
        x = abs_states.cpu().numpy()[:,0]
        y = abs_states.cpu().numpy()[:,1]

        fig = plt.figure(figsize=(5, 3), dpi=200)

        ax = fig.add_subplot(111)
        ax.set_xlabel(r'$X_1$')
        ax.set_ylabel(r'$X_2$')

        for i in range(n):

            predicted1 = learning_algo.transition.predict(torch.cat((abs_states[i:i+1], torch.from_numpy(np.array([[1,0,0,0]])).to(device).float()),-1).to(device)).cpu().detach().numpy()
            predicted2 = learning_algo.transition.predict(torch.cat((abs_states[i:i+1], torch.from_numpy(np.array([[0,1,0,0]])).to(device).float()),-1).to(device)).cpu().detach().numpy()
            predicted3 = learning_algo.transition.predict(torch.cat((abs_states[i:i+1], torch.from_numpy(np.array([[0,0,1,0]])).to(device).float()),-1).to(device)).cpu().detach().numpy()
            predicted4 = learning_algo.transition.predict(torch.cat((abs_states[i:i+1], torch.from_numpy(np.array([[0,0,0,1]])).to(device).float()),-1).to(device)).cpu().detach().numpy()

            ax.plot(np.concatenate([x[i:i+1],predicted1[0,:1]]), np.concatenate([y[i:i+1],predicted1[0,1:2]]), color="0.9", alpha=0.75)
            ax.plot(np.concatenate([x[i:i+1],predicted2[0,:1]]), np.concatenate([y[i:i+1],predicted2[0,1:2]]), color="0.65", alpha=0.75)
            ax.plot(np.concatenate([x[i:i+1],predicted3[0,:1]]), np.concatenate([y[i:i+1],predicted3[0,1:2]]), color="0.4", alpha=0.75)
            ax.plot(np.concatenate([x[i:i+1],predicted4[0,:1]]), np.concatenate([y[i:i+1],predicted4[0,1:2]]), color="0.15", alpha=0.75)

            line3 = ax.scatter(x, y, c="blue", marker='x', edgecolors='k', alpha=0.5, s=50)
            for j, k in enumerate(trajectory):
                ax.text(x[j], y[j], str(k[::-1]), color="black")

        #         test_map = observations[0].unsqueeze(0)
        #         one_hot_right = torch.tensor([[0, 0, 0, 1]], dtype=torch.float).to(device)
        #         one_hot_down = torch.tensor([[0, 1, 0, 0]], dtype=torch.float).to(device)

        #         abstr_test = agent._learning_algo.encoder(test_map)

        #         new_map0 = apply_and_plot(ax, one_hot_right, abstr_test, agent)

        #         print(trajectory[0])
        #         print(agent._learning_algo.Q(abstr_test))
        #         print("[7, 9]")
        #         print(agent._learning_algo.Q(new_map0))

        #         new_map1 = apply_and_plot(ax, one_hot_right, new_map0, agent)
        #         new_map2 = apply_and_plot(ax, one_hot_right, new_map1, agent)
        #         new_map3 = apply_and_plot(ax, one_hot_down, new_map2, agent)

        plt.show()


def apply_and_plot(ax, one_hot_action, abstr_test, agent):
    input_tensor = torch.cat((abstr_test, one_hot_action), dim=-1)
    transition_test = agent._learning_algo.transition(input_tensor)
    ax.scatter(transition_test[0, 0], transition_test[0, 1], c="red", marker='x', edgecolors='k', alpha=0.5, s=100)

    x_test = transition_test.cpu().numpy()[:,0]
    y_test = transition_test.cpu().numpy()[:,1]

    predicted1 = learning_algo.transition.predict(torch.cat((transition_test, torch.from_numpy(np.array([[1,0,0,0]])).to(device).float()),-1).to(device)).cpu().detach().numpy()
    predicted2 = learning_algo.transition.predict(torch.cat((transition_test, torch.from_numpy(np.array([[0,1,0,0]])).to(device).float()),-1).to(device)).cpu().detach().numpy()
    predicted3 = learning_algo.transition.predict(torch.cat((transition_test, torch.from_numpy(np.array([[0,0,1,0]])).to(device).float()),-1).to(device)).cpu().detach().numpy()
    predicted4 = learning_algo.transition.predict(torch.cat((transition_test, torch.from_numpy(np.array([[0,0,0,1]])).to(device).float()),-1).to(device)).cpu().detach().numpy()

    ax.plot(np.concatenate([x_test,predicted1[0,:1]]), np.concatenate([y_test,predicted1[0,1:2]]), color="0.9", alpha=0.75)
    ax.plot(np.concatenate([x_test,predicted2[0,:1]]), np.concatenate([y_test,predicted2[0,1:2]]), color="0.65", alpha=0.75)
    ax.plot(np.concatenate([x_test,predicted3[0,:1]]), np.concatenate([y_test,predicted3[0,1:2]]), color="0.4", alpha=0.75)
    ax.plot(np.concatenate([x_test,predicted4[0,:1]]), np.concatenate([y_test,predicted4[0,1:2]]), color="0.15", alpha=0.75)
    return transition_test



if __name__ == "__main__":
    plt.rcParams.update({'font.size': 5})

    rng = np.random.RandomState(123456)

    experiment_dir = "simple maze novelty reward with 1 step q planning_2019-07-22 13-06-44_17544758"

    root_save_path = os.path.join(ROOT_DIR, "experiments", "maze", "runs")

    experiment_dir = os.path.join(root_save_path, experiment_dir)

    param_fname = os.path.join(experiment_dir, "parameters.json")
    with open(param_fname, 'r') as f:
        parameters = json.load(f)

    parameters = Bunch(parameters)

    score_func = ranked_avg_knn_scores
    if parameters.score_func == "avg_knn_scores":
        score_func = avg_knn_scores

    parameters.score_func = score_func

    knn = batch_knn
    # if parameters.knn == "batch_count_scaled_knn":
    #     knn = batch_count_scaled_knn

    parameters.knn = knn

    env = simple_maze_env(rng,
                          maze_walls=parameters.maze_walls,
                          higher_dim_obs=parameters.higher_dim_obs,
                          size_maze=parameters.size_maze,
                          intern_dim=2)

    # --- Instantiate learning_algo ---
    learning_algo = NSRS(
        env,
        random_state=rng,
        high_int_dim=False,
        **vars(parameters))

    plotter = Plotter(experiment_dir,
                    env_name=parameters.env_name,
                      host='localhost',
                      port=8098,
                      offline=parameters.offline_plotting)

    # network_fname = "model_simple maze novelty_reward_with_d_step_reward_planning_2019-06-18 17-33-07_1347059.epoch=600"
    network_fname = "model.epoch=100"
    network_path = os.path.join(experiment_dir, network_fname)

    dataset_fname = "dataset.epoch=100.pkl"
    dataset_path = os.path.join(experiment_dir, dataset_fname)

    parameters.network_fname = network_path
    parameters.dataset_fname = dataset_path

    agent = SEAgent(
        env,
        learning_algo,
        plotter,
        random_state=rng,
        **vars(parameters))

    # agent._environment._pos_agent = [9, 6] # @ 6, 9
    # step_env(3, agent) # @ 7, 9
    # step_env(3, agent) # @ 8, 9
    # step_env(3, agent) # @ 9, 9

    # agent._environment.summarizePerformance(agent._dataset, agent._learning_algo)



    # plot_obs_with_markings(agent)

    # agent._environment._pos_agent = [19, 8]

    # agent._environment.summarizePerformance(agent._dataset, agent._learning_algo
    agent._environment._pos_agent = [10, 9]

    for m in learning_algo.all_models: m.eval()
    with torch.no_grad():
        original_obs = torch.tensor(agent._environment.observe()[0]).unsqueeze(0).float().to(device)
        encoded_original_obs = learning_algo.encoder(original_obs)
        action = torch.tensor([[0, 1, 0, 0]]).float().to(device)
        transition_input = torch.cat((encoded_original_obs, action), dim=-1)
        encoded_new_obs = learning_algo.transition(transition_input)
        orig_q_vals = learning_algo.Q(encoded_original_obs)
        orig_state_value = orig_q_vals.mean(dim=-1)

        new_q_vals = learning_algo.Q(encoded_new_obs)
        new_state_value = new_q_vals.mean(dim=-1)

        print("here")


    action, score, scores = agent._learning_algo.chooseBestAction(agent._environment.observe(), -1, dataset=agent._dataset,
                                          return_scores=True, action_type=parameters.action_type)
    print("finito")


