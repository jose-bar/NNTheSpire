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
            reward -= damage_taken * 1.5 # Penalize damage taken
            
        if game_state.current_hp > self.combat_starting_hp * 0.9:
            reward += 50
            
        # HP Management
        if self.previous_hp is not None:
            hp_loss = self.previous_hp - game_state.current_hp
            if hp_loss > 0:
                # Higher penalty for losing HP when already low
                hp_percentage = game_state.current_hp / game_state.max_hp
                hp_loss_penalty = -hp_loss * (2 - hp_percentage)
                reward += hp_loss_penalty
                
        # Block Efficiency
        incoming_damage = self.calculate_incoming_damage(game_state)
        if game_state.player.block > 0:
            block_efficiency = min(1.0, game_state.player.block / incoming_damage) if incoming_damage > 0 else 0
            reward += (block_efficiency * 15)  # Max +15 for perfect block
            if game_state.player.block > incoming_damage + 10 and game_state.player.block < incoming_damage + 30:
                reward -= ((game_state.player.block - incoming_damage) * 0.8)  # Penalize overblocking
                
            
        # Penalize wasted energy at turn end
        if not game_state.play_available:  # End of turn
            reward -= (game_state.player.energy * 4)
                
        # Monster Targeting Rewards
        reward += self.calculate_targeting_reward(game_state)
        
        if hasattr(game_state, 'last_potion_used') and game_state.last_potion_used:
            potion_reward = self.calculate_potion_reward(game_state, game_state.last_potion_used)
            reward += potion_reward
        
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
        
        alive_monsters = [m for m in game_state.monsters 
                        if not m.is_gone and not m.half_dead]
        
        # Track monsters that died this turn
        monsters_killed_this_turn = []
        for monster in game_state.monsters:
            if monster.current_hp <= 0 and not monster.is_gone:
                monsters_killed_this_turn.append(monster)
        
        # Reward for kills, scaled by monster threat and HP
        for killed_monster in monsters_killed_this_turn:
            threat_level = self.calculate_threat_level(killed_monster, game_state)
            kill_reward = killed_monster.max_hp * 0.5 * threat_level
            reward += kill_reward
            
        # Penalty for inefficient damage spreading
        low_hp_monsters = [m for m in alive_monsters 
                        if 0 < m.current_hp <= m.max_hp * 0.3]
        if len(low_hp_monsters) > 1:
            reward -= (len(low_hp_monsters) * 15)  # Increased penalty
            
        # Reward for focusing highest threat
        if alive_monsters:
            highest_threat = max(alive_monsters, key=lambda m: self.calculate_threat_level(m, game_state))
            if any(m.current_hp <= 0 for m in game_state.monsters if m == highest_threat):
                reward += 30  # Bonus for killing highest threat
                
        return reward
    
    def calculate_threat_level(self, monster, game_state: Game):
        """Calculate how threatening a monster is"""
        threat = 1.0
        
        # Base threat on damage output
        if monster.move_adjusted_damage:
            damage_threat = (monster.move_adjusted_damage * monster.move_hits) / max(game_state.current_hp, 1)
            threat *= damage_threat
            
        # Additional threat for special intents
        if monster.intent == Intent.BUFF:
            threat *= 1.5
        elif monster.intent == Intent.DEBUFF:
            threat *= 1.3
        elif monster.intent == Intent.STRONG_DEBUFF:
            threat *= 1.8
            
        return threat
    
    def calculate_potion_reward(self, game_state: Game, potion_used):
        """Calculate rewards for potion usage"""
        reward = 0
        
        if not potion_used or not hasattr(potion_used, 'name'):
            return 0
            
        potion_name = potion_used.name.lower()
        
        # Track game state before and after potion use
        pre_state = {
            'player_hp': game_state.current_hp,
            'player_block': game_state.player.block,
            'monster_hp': {i: m.current_hp for i, m in enumerate(game_state.monsters) if not m.is_gone}
        }
        
        # Immediate impact rewards
        if any(x in potion_name for x in ["fire", "explosive", "poison"]):
            # Reward damage dealt, extra reward for kills
            total_damage = sum(
                pre_state['monster_hp'][i] - m.current_hp 
                for i, m in enumerate(game_state.monsters) 
                if i in pre_state['monster_hp']
            )
            reward += total_damage * 0.5
            
            # Extra reward for kills
            kills = sum(
                1 for i, m in enumerate(game_state.monsters)
                if i in pre_state['monster_hp'] and m.current_hp <= 0
            )
            reward += kills * 25
            
        # Defensive potion rewards
        elif any(x in potion_name for x in ["block", "ghost", "bronze"]):
            incoming_damage = self.calculate_incoming_damage(game_state)
            if incoming_damage > 0:
                block_efficiency = min(1.0, game_state.player.block / incoming_damage)
                reward += block_efficiency * 20
                
        # Buff potion rewards
        elif any(x in potion_name for x in ["strength", "dexterity", "focus"]):
            # Higher reward for using early in important fights
            if any(m.max_hp > 100 for m in game_state.monsters):  # Elite/Boss fight
                reward += max(0, 30 - (game_state.turn_count * 3))
                
        # Emergency potion rewards
        elif "fairy" in potion_name:
            if game_state.current_hp < game_state.max_hp * 0.3:
                reward += 15  # Reward for keeping it when low HP
            else:
                reward -= 5   # Small penalty for using when not needed
                
        # Special case rewards
        elif "smoke bomb" in potion_name:
            if any(m.max_hp > 100 for m in game_state.monsters) and game_state.current_hp < game_state.max_hp * 0.3:
                reward += 20  # Big reward for escaping dangerous elite/boss fights when low
            else:
                reward -= 10  # Penalty for using when not necessary
                
        # Card manipulation potion rewards
        elif any(x in potion_name for x in ["skill", "power", "attack", "colorless"]):
            # Reward based on card cost saved
            if hasattr(game_state, 'last_played_card') and game_state.last_played_card:
                reward += game_state.last_played_card.cost * 5
                
        # Energy potion rewards
        elif "energy" in potion_name:
            # Reward based on how much energy was actually used after gaining it
            if hasattr(game_state, 'energy_spent_this_turn'):
                reward += min(game_state.energy_spent_this_turn, 2) * 5
        
        return reward