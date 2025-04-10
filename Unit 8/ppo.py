"""Follows tutorial from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/"""

# Python Utils
import argparse
import os
from distutils.util import strtobool
import time

# Maths
import random
import numpy as np

# Gym
import gym


# PyTorch 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def main():

##############################################
# SETUP
##############################################

    args = parse_args()
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    
    # Set up Tensorboard tracking
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # Set random seed
    random.seed = args.seed
    np.random.seed = args.seed
    torch.manual_seed(args.seed)

    # Set device variables
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create N (num_envs) vectorised environments
    envs = gym.vector.SyncVectorEnv([lambda: make_env(args.env_id, args.seed + i, i, args.record_video, run_name) for i in range(args.num_envs)])

    # Create agent
    agent = Agent(envs).to(device)
    
    # Create PyTorch optimiser
    optimiser = optim.Adam(agent.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    # Initialise variables
    cum_steps_trained = 0
    n = args.num_envs
    m = args.rollout_steps
    s_size = envs.single_observation_space.shape
    a_size = envs.single_action_space.n
    batch_size = n * m
    update_count = int(args.total_timesteps / batch_size)
    next_state = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(n).to(device)

##############################################
# PPO LOOP
##############################################

    # Loop through batches collecting data and then training agent 
    for update in range(update_count):

        # Create storage buffer
        buffer = StorageBuffer(n, m, s_size, a_size, device)#

        # Play the game for m steps, and save data to buffer 
        for i in range(m):
            print(i)
            state = next_state
            with torch.no_grad():
                action, log_prob, _ = agent.act(state)
                value = agent.v(state)
            next_state, reward, done, info = envs.step(action.cpu().numpy())

            # Convert np.ndarray returned by env.step() to Tensor
            next_state = torch.Tensor(next_state).to(device)
            reward = torch.Tensor(reward).to(device)
            done = torch.Tensor(done).to(device)
            
            # Save data to buffer
            buffer.states[i] = next_state
            buffer.actions[i] = action
            buffer.log_probs[i] = log_prob
            buffer.rewards[i] = reward
            buffer.dones[i] = done
            buffer.values[i] = value.flatten()

            cum_steps_trained += n
        
        agent.learn(buffer)

##############################################
# COMMAND LINE ARGUMENTS
##############################################

def parse_args():
    """Parses the command line arguments"""
    parser = argparse.ArgumentParser()
    # Environment variables
    parser.add_argument('--exp-name', type = str, default = os.path.basename(__file__).rstrip(".py"), help = "The experiment name")
    parser.add_argument('--env-id', type = str, default = "LunarLander-v2", help = "The gym environment to be trained")
    
    # Training variables
    parser.add_argument('--lr', type = float, default = 2.5e-4, help = "Learning rate of the optimiser")
    parser.add_argument('--seed', type = int, default = 1, help = "Sets random, numpy, torch seed")
    parser.add_argument('--total-timesteps', type = int, default = 25000, help = "Total timesteps of the experiment")
    
    # Device variables
    parser.add_argument('--torch-deterministic', type = lambda x: bool(strtobool(x)), default = True, nargs = '?', const = True, help = "If toggled, torch.backends.cudnn.deterministic=False")
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help="If toggled, cuda will not be enabled")
    
    # Video recording variable
    parser.add_argument('--record-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help="Saves videos of the agent performance to the videos folder")

    # PPO variables
    parser.add_argument('--num-envs', type=int, default=4, help="The number of vectorised environments")
    parser.add_argument('--rollout-steps', type=int, default=100, help="The number of steps per rollout per environment")
    parser.add_argument('--weight-decay', type=int, default=1e-4, help="The weight decay / L2 regularisation for the adam optimiser")

    args = parser.parse_args()
    return args


##############################################
# ACTOR AND CRITIC NETWORKS
##############################################

class Agent(nn.Module):
    """Actor and critic networks"""
    def __init__(self, envs):
        super(Agent, self).__init__()
        # Shared feature network
        self.hidden = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        
        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def v(self, x):
        """Returns the estimated value of state x according to the critic network"""
        return self.critic(self.hidden(x))
    

    def act(self, x, action = None):
        """Takes a state and return a tuple of:
        action selected at random from the probability distribution produced by the actor network,
        the log_prob of action,
        the entropy of the action probability distribution"""
        probs = self.actor(self.hidden(x))
        cat = Categorical(probs)
        if action == None:
            action = cat.sample()
        return action, cat.log_prob(action), cat.entropy()


    def learn(self, buffer):
        raise NotImplementedError

##############################################
# STORAGE BUFFER
##############################################

class StorageBuffer():
    def __init__(self, n, m, s_size, a_size, device):
        self.states = torch.zeros((m, n) + s_size).to(device)
        self.actions = torch.zeros((m, n)).to(device)
        self.log_probs = torch.zeros((m, n)).to(device)
        self.rewards = torch.zeros((m, n)).to(device)
        self.dones = torch.zeros((m, n)).to(device)
        self.values = torch.zeros((m, n)).to(device)
 

##############################################
# HELPER FUNCTIONS
##############################################

def make_env(env_id, seed, env_num, record_video, run_name):
    """Creates a single gym environment"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Record video for first environment, if requested
    if record_video:
        if env_num == 0:
            trigger = lambda t: t % 100 == 0
            env = gym.wrappers.RecordVideo(env, f"./videos/{run_name}", episode_trigger = trigger)
    
    # Set seed
    env.seed = seed
    env.action_space.seed = seed
    env.observation_space.seed = seed
    
    return env


def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    """Initialises layer weights and biases"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


if __name__ == "__main__":
    main()