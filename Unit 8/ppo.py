"""Follows tutorial from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/"""

# Python Utils
import argparse
import os
from distutils.util import strtobool
import time
from math import exp

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

    # Parse command line arguments
    args = parse_args()
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    
    # Set up Tensorboard tracking (view using: !tensorboard --logdir runs)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
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
    batch_size = n * m
    minibatch_count = args.minibatches
    epochs = args.epochs
    gamma = args.gamma
    clip_param = args.clip_param
    ent_coef = args.ent_coef
    vl_coef = args.vl_coef
    norm_advantages = args.norm_advantages
    gae_lambda = args.gae_lambda
    update_count = int(args.total_timesteps // batch_size)
    next_state = torch.Tensor(envs.reset()).float().to(device)
    done = torch.zeros(n).to(device)

##############################################
# PPO LOOP
##############################################

    # Loop through batches collecting data and then training agent 
    for update in range(update_count):

        # Create storage buffer
        buffer = StorageBuffer(n, m, s_size, device)

        # Play the game for m steps, and save data to buffer 
        for i in range(m):
            # Save state and done to buffer
            state = next_state
            buffer.states[i] = state
            buffer.dones[i] = done
            
            # Get action and take an environment step
            with torch.no_grad():
                action, log_prob, _ = agent.act(state)
                value = agent.v(state)
            next_state, reward, done, info = envs.step(action.cpu().numpy())

            # Convert np.ndarray returned by env.step() to Tensor
            next_state = torch.Tensor(next_state).float().to(device)
            reward = torch.Tensor(reward).float().to(device)
            done = torch.Tensor(done).float().to(device)
            
            # Save action, action_log prob, reward and estimated value to buffer
            buffer.actions[i] = action
            buffer.log_probs[i] = log_prob
            buffer.rewards[i] = reward
            buffer.values[i] = value.flatten()

            cum_steps_trained += n
            
            # Log returns and episode lengths into tensorboard
            for item in info:
                if "episode" in item.keys():
                    print(f"cumulative_steps_trained={cum_steps_trained}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], cum_steps_trained)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], cum_steps_trained)
                    break
        
        # Calculate advantages and returns using:
        # a(t) = r(t) + gamma*v(t+1) - v(t), using v() from critic network
        # ret(t) = sum(r(t) + gamma * r(t+1) + .... gamma^k * r(t+k))
        with torch.no_grad():
            next_value = agent.v(next_state).flatten()
            gae = 0
            for i in reversed(range(m)):
                if i == m - 1:
                    nextnonterminal = 1.0 - done
                    next_values = next_value
                else:
                    nextnonterminal = 1.0 - buffer.dones[i + 1]
                    next_values = buffer.values[i + 1]

                delta = buffer.rewards[i] + gamma * next_values * nextnonterminal - buffer.values[i]
                gae = delta + gamma * gae_lambda * nextnonterminal * gae
                buffer.advantages[i] = gae
                buffer.returns[i] = buffer.advantages[i] + buffer.values[i]

        # Train agent using sgd/backprop
        agent.learn(optimiser, buffer, epochs, s_size, batch_size, minibatch_count, clip_param, ent_coef, vl_coef, norm_advantages, writer, cum_steps_trained)

##############################################
# COMMAND LINE ARGUMENTS
##############################################

def parse_args():
    """Parses the command line arguments"""
    parser = argparse.ArgumentParser()
    # Environment variables
    parser.add_argument('--exp-name', type = str, default = os.path.basename(__file__).rstrip(".py"), help = "The experiment name")
    parser.add_argument('--env-id', type = str, default = "LunarLander-v2", help = "The gym environment to be trained")
    parser.add_argument('--seed', type = int, default = 1, help = "Sets random, numpy, torch seed")
    
    # Device variables
    parser.add_argument('--torch-deterministic', type = lambda x: bool(strtobool(x)), default = True, nargs = '?', const = True, help = "If toggled, torch.backends.cudnn.deterministic=False")
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help="If toggled, cuda will not be enabled")
    
    # Video recording variable
    parser.add_argument('--record-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help="Saves videos of the agent performance to the videos folder")

    # Training variables
    parser.add_argument('--total-timesteps', type = int, default = 25000, help = "Total timesteps of the experiment")
    parser.add_argument('--num-envs', type=int, default=4, help="The number of vectorised environments")
    parser.add_argument('--rollout-steps', type=int, default=500, help="The number of steps per rollout per environment")
    parser.add_argument('--epochs', type=int, default=4, help="The number of epochs to train in each update")
    parser.add_argument('--minibatches', type=int, default=4, help="The number of minibatches")
    parser.add_argument('--lr', type = float, default = 2.5e-4, help = "Learning rate of the optimiser")
    parser.add_argument('--gamma', type=float, default=0.995, help="The discount rate for returns")
    parser.add_argument('--clip-param', type=float, default=0.2, help="The clip coefficient for the clipped surrogate objective function")
    parser.add_argument('--weight-decay', type=float, default=1e-4, help="The weight decay / L2 regularisation for the adam optimiser")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy loss")
    parser.add_argument("--vl-coef", type=float, default=0.5, help="coefficient of the value loss")
    parser.add_argument('--norm-advantages', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help="Normalises minibatch advantages")
    parser.add_argument('--gae-lambda', type=float, default=0.95, help="Lambda for Generalized Advantage Estimation")
    
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
        x = x.float()
        return self.critic(self.hidden(x))
    

    def act(self, x, action = None):
        """Takes a state and return a tuple of:
        action selected at random from the probability distribution produced by the actor network,
        the log_prob of action,
        the entropy of the action probability distribution"""
        x = x.float()
        probs = self.actor(self.hidden(x))
        cat = Categorical(probs)
        if action == None:
            action = cat.sample()
        return action, cat.log_prob(action), cat.entropy()


    def learn(self, optimiser, buffer, epochs, s_size, batch_size, minibatch_count, clip_param, ent_coef, vl_coef, norm_advantages, writer, cum_steps_trained):        
        # Flatten vectorised environments
        states = buffer.states.reshape((-1,) + s_size)
        actions = buffer.actions.reshape(-1)
        log_probs = buffer.log_probs.reshape(-1)
        returns = buffer.returns.reshape(-1)
        advantages = buffer.advantages.reshape(-1)
        values = buffer.values.reshape(-1)

        # Train for x epochs
        minibatch_size = batch_size // minibatch_count
        for i in range(epochs):
            # Shuffle data
            batch_inds = np.arange(batch_size)
            np.random.shuffle(batch_inds)

            # Split into minibatches
            minibatches = []
            for i in range(0, batch_size, minibatch_size):
                minibatches.append(batch_inds[i:i + minibatch_size])

            # Train minibatches
            for minibatch in minibatches:
                # Calculate policy loss
                mb_old_log_probs = log_probs[minibatch]
                _, mb_new_log_probs, mb_entropys = self.act(states[minibatch],actions[minibatch])
                mb_log_ratios = mb_new_log_probs - mb_old_log_probs
                mb_ratios = mb_log_ratios.exp()
                mb_advantages = advantages[minibatch]
                if norm_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                policy_loss = torch.max(-mb_advantages * mb_ratios, -mb_advantages * torch.clamp(mb_ratios, 1 - clip_param, 1 + clip_param)).mean()
                
                # Calculate entropy loss
                entropy_loss = mb_entropys.mean()

                # Calculate value loss
                mb_new_values = self.v(states[minibatch]).flatten()
                clipped_values = values[minibatch] + torch.clamp(mb_new_values - values[minibatch], -clip_param, clip_param)
                value_loss = 0.5 * torch.max((mb_new_values - returns[minibatch]) ** 2,(clipped_values - returns[minibatch]) ** 2).mean()

                # Total loss
                ppo_loss = policy_loss - ent_coef * entropy_loss + vl_coef * value_loss

                # Backpropagation
                optimiser.zero_grad()
                ppo_loss.backward()
                optimiser.step()

        writer.add_scalar("loss/policy", policy_loss.item(), cum_steps_trained)
        writer.add_scalar("loss/value", value_loss.item(), cum_steps_trained)
        writer.add_scalar("loss/entropy", entropy_loss.item(), cum_steps_trained)


##############################################
# STORAGE BUFFER
##############################################

class StorageBuffer():
    def __init__(self, n, m, s_size, device):
        self.states = torch.zeros((m, n) + s_size).to(device)
        self.actions = torch.zeros((m, n), dtype=torch.long).to(device)
        self.log_probs = torch.zeros((m, n)).to(device)
        self.rewards = torch.zeros((m, n)).to(device)
        self.dones = torch.zeros((m, n)).to(device)
        self.values = torch.zeros((m, n)).to(device)
        self.advantages = torch.zeros((m, n)).to(device)
        self.returns = torch.zeros((m, n)).to(device)


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