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

class SlayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SlayerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    from neuralAgent import NeuralAgent
    # Initialize agent and coordinator
    agent = NeuralAgent()
    coordinator = Coordinator()
    
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
        
        # Optimize model - Fix this line
        agent.optimize_model()  # Remove the extra agent parameter
        
        # Log progress
        # print(f"Game {game_number + 1}:")
        # print(f"  Victory: {result}")
        # print(f"  Total damage taken: {agent.total_damage_taken}")
        # print(f"  Win rate: {agent.victories/(game_number+1)*100:.2f}%")
        # print(f"  Epsilon: {agent.epsilon:.3f}")
        