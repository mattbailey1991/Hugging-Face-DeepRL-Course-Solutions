"""Loosely follows tutorial from https://www.youtube.com/embed/MEt6rrxH8W4"""

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

# PyTorch / Weights & Biases 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def main():
    args = parse_args()
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    
    # Set up Weights & Bias tracking 
    if args.track:
        import wandb

        wandb.init(
            project = args.wandb_project_name,
            entity = args.wandb_team_name,
            sync_tensorboard = True,
            config=vars(args),
            name = run_name,
            monitor_gym = True,
            save_code = True 
        )

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

    # Create vectorised environments
    envs = gym.vector.SyncVectorEnv([lambda: make_env(args.env_id, args.seed + i, i, args.record_video, run_name) for i in range(args.num_envs)])

    # Create agent
    agent = Agent(envs).to(device)
    
    # Create PyTorch optimiser
    optimiser = optim.Adam(agent.parameters, lr = args.learning_rate, weight_decay = args.weight_decay)


def parse_args():
    """Parses the command line arguments"""
    parser = argparse.ArgumentParser()
    # Environment variables
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"), help="The experiment name")
    parser.add_argument('--env-id', type=str, default="LunarLander-v2", help="The gym environment to be trained")
    
    # Training variables
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help="Learning rate of the optimiser")
    parser.add_argument('--seed', type=int, default=1, help="Sets random, numpy, torch seed")
    parser.add_argument('--total-timesteps', type=int, default=25000, help="Total timesteps of the experiment")
    
    # Device variables
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help="If toggled, torch.backends.cudnn.deterministic=False")
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help="If toggled, cuda will not be enabled")
    
    # Weights and biases variables
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help="If toggled, the project will be tracked with weights and biases")
    parser.add_argument('--wandb-project-name', type=str, default="Test Project", help="Weights and biases project name")
    parser.add_argument('--wandb-team-name', type=str, default="mattbailey1991-study", help="Weights and biases team name")

    # Video recording variable
    parser.add_argument('--record-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help="Saves videos of the agent performance to the videos folder")

    # PPO variables
    parser.add_argument('--num-envs', type=int, default=4, help="The number of vectorised environments")
    parser.add_argument('--weight-decay', type=int, default=1e-4, help="The weight decay / L2 regularisation for the adam optimiser")

    args = parser.parse_args()
    return args


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
            nn.Softmax()
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
        m = Categorical(probs)
        if action == None:
            action = m.sample()
        return action, m.log_prob(action), m.entropy()


class StorageBuffer():
    def hello():
        raise NotImplementedError


def make_env(env_id, seed, env_num, record_video, run_name):
    """Creates a single gym environment"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if record_video:
        # First environment only
        if env_num == 0:
            trigger = lambda t: t % 100 == 0
            env = gym.wrappers.RecordVideo(env, f"./videos/{run_name}", episode_trigger = trigger)
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