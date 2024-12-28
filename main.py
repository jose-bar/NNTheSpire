import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
import random

from spirecomm.spire.screen import RestOption
from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass
from spirecomm.spire.game import Game
from spirecomm.communication.action import *
from checkpointHandler import CheckpointHandler

class SlayerNet(nn.Module):
    def __init__(self, input_size=32, hidden_size=32, lstm_layers=2, output_size=32):
        super(SlayerNet, self).__init__()
        
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
        # Reduce input dimensions with initial linear layer
        self.input_reducer = nn.Linear(input_size, hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Output layers
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # If input is not 3D, add sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        # Reduce input dimensions
        x = self.input_reducer(x)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
            
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last output for decision making
        last_output = lstm_out[:, -1, :]
        
        # Dueling DQN architecture
        advantage = self.advantage(last_output)
        value = self.value(last_output)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values, hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)

if __name__ == "__main__":
    from neuralAgent import NeuralAgent
    from logger import GameStateLogger
    
    # Initialize agent and coordinator
    agent = NeuralAgent()  # Updated input size
    coordinator = Coordinator()
    checkpoint_handler = CheckpointHandler()
    
    # Try to load latest checkpoint if it exists
    if checkpoint_handler.load_latest_checkpoint(agent):
        GameStateLogger.log_error("We stinky with it")
    
    # Setup coordinator callbacks
    coordinator.signal_ready()
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)
    
    # Training loop
    for game_number in range(1000):  # Train for 1000 games
        chosen_class = random.choice(list(PlayerClass))
        agent.change_class(chosen_class)
        
        # Reset episode-specific tracking
        agent.previous_hp = None
        agent.combat_starting_hp = None
        agent.in_combat_previously = False
        
        # Decay exploration rate
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        # Play one game
        result = coordinator.play_one_game(chosen_class)
        
        if result:
            agent.victories += 1
        
        # Optimize model
        agent.optimize_model()
        
        # Save checkpoint every 10 games
        if (game_number + 1) % 10 == 0:
            stats = {
                'win_rate': agent.victories / (game_number + 1),
                'avg_damage_taken': agent.total_damage_taken / (game_number + 1),
                'epsilon': agent.epsilon
            }
            checkpoint_handler.save_checkpoint(agent, game_number + 1, stats)