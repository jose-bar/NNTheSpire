from spirecomm.spire.game import Game
from spirecomm.spire.character import Intent

class RewardHandler:
    def __init__(self):
        # Initialize tracking variables
        self.combat_starting_hp = None
        self.previous_hp = None
        self.in_combat_previously = False
        
    def calculate_reward(self, game_state: Game):
        """Main reward calculation function that combines all reward components"""
        total_reward = 0
        
        # Combat-specific rewards
        if game_state.in_combat:
            total_reward += self.calculate_combat_reward(game_state)
        else:
            total_reward += self.calculate_non_combat_reward(game_state)
            
        # Update state tracking
        self.previous_hp = game_state.current_hp
        self.in_combat_previously = game_state.in_combat
        
        return total_reward
    
    def calculate_combat_reward(self, game_state: Game):
        """Calculate rewards for in-combat decisions"""
        reward = 0
        
        # Track combat entry/exit
        if game_state.in_combat and not self.in_combat_previously:
            self.combat_starting_hp = game_state.current_hp
        elif not game_state.in_combat and self.in_combat_previously:
            damage_taken = self.combat_starting_hp - game_state.current_hp
            reward -= damage_taken  # Penalize damage taken
            reward += max(0, 50 - damage_taken)  # Bonus for minimizing damage
            
        # HP Management
        if self.previous_hp is not None:
            hp_loss = self.previous_hp - game_state.current_hp
            if hp_loss > 0:
                # Higher penalty for losing HP when already low
                hp_percentage = game_state.current_hp / game_state.max_hp
                hp_loss_penalty = -hp_loss * (1.5 - hp_percentage)
                reward += hp_loss_penalty
                
        # Block Efficiency
        incoming_damage = self.calculate_incoming_damage(game_state)
        if game_state.player.block > 0:
            block_efficiency = min(1.0, game_state.player.block / incoming_damage)
            reward += (block_efficiency * 10)  # Max +10 for perfect block
            if game_state.player.block > incoming_damage + 10:
                reward -= ((game_state.player.block - incoming_damage) * 0.5)  # Penalize overblocking
                
        # Energy Efficiency
        if game_state.play_available:
            playable_cards = [card for card in game_state.hand if card.is_playable]
            zero_cost_cards = [card for card in playable_cards if card.cost == 0]
            if len(zero_cost_cards) > 0:
                reward += 2  # Reward playing zero-cost cards
            
            # Penalize wasted energy at turn end
            if not game_state.play_available:  # End of turn
                reward -= (game_state.player.energy * 3)
                
        # Monster Targeting Rewards
        reward += self.calculate_targeting_reward(game_state)
        
        return reward
    
    def calculate_non_combat_reward(self, game_state: Game):
        """Calculate rewards for out-of-combat decisions"""
        reward = 0
        
        # Rest site decisions
        if game_state.room_type == "RestRoom":
            if game_state.current_hp < game_state.max_hp / 3:
                reward += 30  # Encourage resting when low HP
            elif game_state.act != 1 and game_state.floor % 17 == 15:  # Pre-boss rest site
                if game_state.current_hp < game_state.max_hp * 0.9:
                    reward += 20  # Encourage resting before boss if not near full HP
                    
        # Shop rewards
        elif game_state.room_type == "ShopRoom":
            if game_state.gold >= 80:
                reward += 10  # Encourage shop visits with good gold
                
        # Deck size penalties
        deck_size = len(game_state.deck)
        if deck_size > 35:
            reward -= (deck_size - 35) * 2  # Penalty for oversized deck
            
        return reward
    
    def calculate_incoming_damage(self, game_state: Game):
        """Calculate expected incoming damage from monsters"""
        total_damage = 0
        for monster in game_state.monsters:
            if not monster.is_gone and not monster.half_dead:
                if monster.move_adjusted_damage:
                    total_damage += monster.move_adjusted_damage * monster.move_hits
                else:
                    # Default damage assumption for unknown intents
                    total_damage += 5 * game_state.act
        return total_damage
    
    def calculate_targeting_reward(self, game_state: Game):
        """Calculate rewards for strategic monster targeting"""
        reward = 0
        
        # Reward for focusing on threatening enemies
        alive_monsters = [m for m in game_state.monsters 
                         if not m.is_gone and not m.half_dead]
        
        if len(alive_monsters) > 1:
            # Reward for having monsters near death
            low_hp_monsters = [m for m in alive_monsters 
                             if m.current_hp <= m.max_hp * 0.3]
            reward += (len(low_hp_monsters) * 15)
            
            # AOE value
            reward += 5  # Base reward for AOE opportunity
            
        # Assess monster threats
        for monster in alive_monsters:
            threat_level = self.calculate_threat_level(monster, game_state)
            if monster.current_hp <= 0:
                reward += threat_level * 20  # Reward for killing threats
                
        return reward
    
    def calculate_threat_level(self, monster, game_state: Game):
        """Calculate how threatening a monster is"""
        threat = 1.0
        
        # Base threat on damage output
        if monster.move_adjusted_damage:
            damage_threat = (monster.move_adjusted_damage * monster.move_hits) / game_state.current_hp
            threat *= damage_threat
            
        # Additional threat for special intents
        if monster.intent == Intent.BUFF:
            threat *= 1.5
        elif monster.intent == Intent.DEBUFF:
            threat *= 1.3
        elif monster.intent == Intent.STRONG_DEBUFF:
            threat *= 1.8
            
        return threat