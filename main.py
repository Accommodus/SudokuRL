import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
from itertools import count
import math
from SudokuEnv import SudokuEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    """
    A Deep Q-Network (DQN) model with three fully connected layers.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer, output layer.
    """

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    A simple memory buffer for storing transitions.

    Attributes:
        memory (deque): A double-ended queue to store transitions.
        capacity (int): The maximum size of the memory buffer.
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions from memory.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            list: A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the current size of the memory."""
        return len(self.memory)


def select_action(state, steps_done, policy_net, n_actions, EPS_START, EPS_END, EPS_DECAY):
    """
    Selects an action based on the current policy and exploration rate.

    Args:
        state (torch.Tensor): The current state of the environment.
        steps_done (int): The number of steps completed so far.
        policy_net (DQN): The current DQN policy network.
        n_actions (int): The number of possible actions.
        EPS_START (float): The starting value of the exploration rate.
        EPS_END (float): The minimum value of the exploration rate.
        EPS_DECAY (float): The rate of decay for exploration rate.

    Returns:
        torch.Tensor: The chosen action.
    """
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(memory, BATCH_SIZE, policy_net, target_net, optimizer, GAMMA):
    """
    Performs a single step of optimization on the policy network.

    Args:
        memory (ReplayMemory): The replay memory buffer.
        BATCH_SIZE (int): The batch size for optimization.
        policy_net (DQN): The current DQN policy network.
        target_net (DQN): The target DQN network.
        optimizer (torch.optim): The optimizer for the policy network.
        GAMMA (float): The discount factor for future rewards.
    """
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
LR = 1e-4

env = SudokuEnv()
n_observations = env.observation_space.shape[0] * env.observation_space.shape[1]
n_actions = env.row_length * env.row_length * 9

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)

steps_done = 0

num_episodes = 50
for i_episode in range(num_episodes):
    state = env.reset()
    state = torch.from_numpy(state).float().view(-1).to(device).unsqueeze(0)

    for t in count():
        # Determine the number of valid actions for the current state
        n_actions = len([(i, j) for i in range(env.row_length) for j in range(env.row_length) if env.sudoku.board[i][j] == 0]) * 9
        action_idx = select_action(state, steps_done, policy_net, n_actions, EPS_START, EPS_END, EPS_DECAY).item()
        action_tuple = env.map_action_to_cell(action_idx)

        # Perform a step in the environment
        next_state, reward, done, _ = env.step(action_tuple)
        reward_tensor = torch.tensor([reward], device=device)

        # Flatten the next state if it's not terminal
        if not done:
            next_state_tensor = torch.from_numpy(next_state).float().view(-1).to(device).unsqueeze(0)
        else:
            next_state_tensor = None

        # Store the transition in memory
        memory.push(state, torch.tensor([[action_idx]], device=device), next_state_tensor, reward_tensor)

        # Update the current state
        state = next_state_tensor if next_state_tensor is not None else state

        # Perform one step of optimization
        optimize_model(memory, BATCH_SIZE, policy_net, target_net, optimizer, GAMMA)

        if done:
            break

    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Training complete')

