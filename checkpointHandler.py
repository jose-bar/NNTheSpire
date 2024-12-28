import os
import torch
import json
from datetime import datetime

class CheckpointHandler:
    def __init__(self, base_dir='checkpoints'):
        self.base_dir = base_dir
        self.stats_file = os.path.join(base_dir, 'training_stats.json')
        os.makedirs(base_dir, exist_ok=True)
        
        # Load existing stats if they exist
        self.training_stats = self._load_stats()
        
    def _load_stats(self):
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        return {
            'total_games': 0,
            'win_rates': [],
            'avg_floors_reached': [],
            'checkpoints': []
        }
        
    def save_checkpoint(self, agent, game_number, additional_stats=None):
        """Save a checkpoint of the model and training stats"""
        # Create checkpoint filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(self.base_dir, f'checkpoint_{game_number}_{timestamp}.pt')
        
        # Prepare checkpoint data
        checkpoint = {
            'game_number': game_number,
            'policy_net_state': agent.policy_net.state_dict(),
            'target_net_state': agent.target_net.state_dict(),
            'optimizer_state': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
        }
        
        # Save model checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update training stats
        self.training_stats['total_games'] = game_number
        self.training_stats['checkpoints'].append({
            'game_number': game_number,
            'timestamp': timestamp,
            'path': checkpoint_path
        })
        
        # Add additional stats if provided
        if additional_stats:
            for key, value in additional_stats.items():
                if key not in self.training_stats:
                    self.training_stats[key] = []
                self.training_stats[key].append(value)
        
        # Save updated stats
        with open(self.stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
            
    def load_latest_checkpoint(self, agent):
        """Load the most recent checkpoint"""
        if not self.training_stats['checkpoints']:
            return False
            
        # Get latest checkpoint
        latest = max(self.training_stats['checkpoints'], 
                    key=lambda x: x['game_number'])
        
        if os.path.exists(latest['path']):
            checkpoint = torch.load(latest['path'])
            
            # Load model states
            agent.policy_net.load_state_dict(checkpoint['policy_net_state'])
            agent.target_net.load_state_dict(checkpoint['target_net_state'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
            agent.epsilon = checkpoint['epsilon']
            
            return True
        return False
        
    def get_training_summary(self):
        """Get a summary of training progress"""
        if not self.training_stats['checkpoints']:
            return "No training data available."
            
        latest = self.training_stats['checkpoints'][-1]
        recent_win_rate = self.training_stats['win_rates'][-10:] if 'win_rates' in self.training_stats else []
        
        summary = f"Games played: {self.training_stats['total_games']}\n"
        summary += f"Latest checkpoint: Game {latest['game_number']}\n"
        if recent_win_rate:
            summary += f"Recent win rate: {sum(recent_win_rate)/len(recent_win_rate):.2%}\n"
        
        return summary