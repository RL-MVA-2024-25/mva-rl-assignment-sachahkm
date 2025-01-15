import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
from gymnasium.wrappers import TimeLimit
import os 


# Replay buffer to save the transitions (s, a, r, s', done) ==> reduce correlation 
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.idx = 0

    # Add a transition to the buffer 
    def append(self, s, a, r, s_new, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = (s, a, r, s_new, done)
        self.idx = (self.idx + 1) % self.capacity

    # Sample a batch in the buffer
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        local_fct = lambda x: torch.Tensor(np.array(x)).to(self.device)
        return [local_fct(x) for x in list(zip(*batch))]

    # Special method to compute the buffer length
    def __len__(self):
        return len(self.buffer)

# Neural network to estimate Q (Deep-Q-Network)
class DQN(nn.Module):
    def __init__(self, hidden_size, nb_hidden):
        super(DQN, self).__init__()
        self.in_layer = nn.Linear(6, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)]*(nb_hidden))
        self.out_layer = nn.Linear(hidden_size, 4)

    # Forward function (apply the neural network to the input x)
    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.out_layer(x)

# DQN agent 
class ProjectAgent:
    def __init__(self):
        # Device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Buffer parameters
        self.batch_size = 512
        self.buffer_capacity = int(1e6)
        self.memory = ReplayBuffer(self.buffer_capacity, self.device)

        # DQN parameters
        self.hidden_size = 256
        self.nb_hidden = 5
        self.model = DQN(self.hidden_size, self.nb_hidden).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)

        # Eps-greedy parameters
        self.epsilon_min = 0.1
        self.epsilon_max = 1.0
        self.epsilon_decay_period = 1500
        self.epsilon_delay_decay = 20
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_period

        # DQN agent parameters 
        self.gamma = 0.95
        self.criterion = nn.SmoothL1Loss()
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.update_target_freq = 20
        self.update_target_tau = 0.005

        # Model to save
        self.best_score = -np.inf
        self.best_score_dr = -np.inf
        self.best_weighted_score = -np.inf 
        self.selected_model = 'best_score.pth'

    # Greedy action: maximize Q(s, .)
    def greedy_action(self, network, s):
        with torch.no_grad():
            Q = network(torch.Tensor(s).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

    # Agent policy 
    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            return self.greedy_action(self.model, observation)

    # Backpropagation 
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            s_batch, a_batch, r_batch, s_new_batch, done_batch = self.memory.sample(self.batch_size)
            Q_new_max = self.target_model(s_new_batch).max(1)[0].detach()
            update = torch.addcmul(r_batch, 1 - done_batch, Q_new_max, value=self.gamma)
            Q_values = self.model(s_batch).gather(1, a_batch.to(torch.long).unsqueeze(1))
            loss = self.criterion(Q_values, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # Agent training
    def train(self, max_episode):
        global env 

        # Initialization 
        episode = 1
        step = 0
        epsilon = self.epsilon_max

        # Iterative update of the agent 
        while episode <= max_episode:
            # Alternate training on single patient and population
            if (episode - 1) % 10 == 0:
                env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)   
            if (episode - 1) % 10 == 5:
                env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)  

            # Initialization of the new episode  
            s, _ = env.reset()
            total_reward = 0
            done = False
            trunc = False

            # Interact with the environement until the end of episode   
            while not (done or trunc):
                # Decrease epsilon at each new step of the episode 
                if step >= self.epsilon_delay_decay:
                    epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

                # Eps-greedy action 
                if np.random.rand() < epsilon:
                    a = self.act(s, use_random=True)
                else:
                    a = self.act(s, use_random=False)

                # Observe a transition 
                s_new, r, done, trunc, _ = env.step(a)
                self.memory.append(s, a, r, s_new, done)
                total_reward += r

                # Update the DQN
                self.gradient_step()

                # Update the target 
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                
                # Prepare for the next transition 
                s = s_new
                step += 1

            # Display the total reward of the episode 
            print(f"Episode {episode}, Total reward: {total_reward:.2e}")

            # Compute different scores 
            score = evaluate_HIV(agent=self, nb_episode=5)
            score_dr = evaluate_HIV_population(agent=self, nb_episode=10)
            weighted_score = (2 * score + score_dr) / 3

            # Display and save the model if there is score improvement 
            if score > self.best_score:
                self.best_score = score 
                print(f'New best score: {score:.2e}')
                self.save('best_score.pth')
            if score_dr > self.best_score_dr:
                self.best_score_dr = score_dr 
                print(f'New best score_dr: {score_dr:.2e}')
                self.save('best_score_dr.pth')
            if weighted_score > self.best_weighted_score:
                self.best_weighted_score = weighted_score 
                print(f'New best weighted_score: {weighted_score:.2e}')
                self.save('best_weighted_score.pth')

            # Prepare for the next episode 
            episode += 1

    # Save the agent
    def save(self, path):
        # Create the folder "agents" if it doesn't exist 
        folder = 'agents'
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save our agent there 
        full_path = os.path.join(folder, path)
        torch.save(self.model.state_dict(), full_path)

    # Load the agent 
    def load(self):
        full_path = os.path.join('agents', self.selected_model)
        self.model.load_state_dict(torch.load(full_path, map_location=self.device))
        self.model.eval()

# Train the agent 
if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train(max_episode=5000)
