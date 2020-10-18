import numpy as np

from torch.utils.data import DataLoader
from random import sample

from nsrl.helper.pytorch import DataSet
from simple_maze_env_pytorch import MyEnv as simple_maze_env

if __name__ == "__main__":

    rng = np.random.RandomState()
    env = simple_maze_env(rng,
                      maze_walls=False,
                      higher_dim_obs=False,
                      size_maze=21,
                      intern_dim=2)

    dataset = DataSet(env)

    env.reset(-1)
    num_steps = 128

    actions = list(range(env.nActions()))

    for i in range(num_steps):
        obs = env.observe()
        action = sample(actions, 1)[0]
        reward = env.act(action)
        is_terminal = env.inTerminalState()
        dataset.addSample(obs, action, reward, is_terminal, 0)

    state, action, reward, next_state, terminal, idx = dataset[3]

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    for i, batch in enumerate(dataloader):
        print(batch)

    print("done")

