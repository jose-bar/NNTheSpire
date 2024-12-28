from spirecomm.spire.game import Game
from spirecomm.communication.action import *
from spirecomm.spire.character import Intent
from spirecomm.spire.card import *
from logger import GameStateLogger
import torch
import torch.nn.functional as F

class DecisionHandler:
    def __init__(self, policy_net, input_size=32):
        self.policy_net = policy_net
        self.input_size = input_size
        self.logger = GameStateLogger()
    
    def evaluate_cards(self, cards, game_state: Game):
        """Evaluate a list of cards using the neural network"""
        # Get all current card values
        card_values = []
        
        for card in cards:
            # Get card features
            features = self.get_card_features(card, game_state)
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
            state_tensor = state_tensor.to(next(self.policy_net.parameters()).device)
            
            # Get neural network evaluation
            with torch.no_grad():
                q_values, _ = self.policy_net(state_tensor)
                card_values.append(q_values.mean().item())
        
        return torch.tensor(card_values)
    
    def get_card_features(self, card, game_state: Game):
        """Get features for a single card considering game context"""
        features = []
        
        # Basic card properties (4 features)
        features.extend([
            card.cost / 3,  # Normalized cost
            1 if card.is_playable else 0,
            1 if card.exhausts else 0,
            1 if card.has_target else 0
        ])
        
        # Card type (3 features)
        features.extend([
            1 if card.type == CardType.ATTACK else 0,
            1 if card.type == CardType.SKILL else 0,
            1 if card.type == CardType.POWER else 0
        ])
        
        # Card synergies with deck (5 features)
        deck_cards = game_state.deck if hasattr(game_state, 'deck') else []
        features.extend([
            # Strength synergy
            1 if self._has_strength_synergy(card, deck_cards) else 0,
            
            # Block synergy
            1 if self._has_block_synergy(card, deck_cards) else 0,
            
            # Exhaust synergy
            1 if self._has_exhaust_synergy(card, deck_cards) else 0,
            
            # Strike synergy
            1 if self._has_strike_synergy(card, deck_cards) else 0,
            
            # Self damage synergy
            1 if self._has_self_damage_synergy(card, deck_cards) else 0
        ])
        
        #GameStateLogger().log_game_state(game_state)
        
        # Game state context (8 features)
        features.extend([
            game_state.current_hp / game_state.max_hp,  # HP ratio
            len(deck_cards) / 40,  # Deck size
            game_state.act / 3,  # Current act
            len([c for c in deck_cards if c.type == CardType.ATTACK]) / max(1, len(deck_cards)),  # Attack ratio
            len([c for c in deck_cards if c.type == CardType.SKILL]) / max(1, len(deck_cards)),   # Skill ratio
            len([c for c in deck_cards if c.type == CardType.POWER]) / max(1, len(deck_cards)),   # Power ratio
            len([c for c in deck_cards if c.exhausts]) / max(1, len(deck_cards)),                 # Exhaust ratio
            len([c for c in deck_cards if c.cost == 0]) / max(1, len(deck_cards))                 # Zero cost ratio
        ])
        
        while len(features) < self.input_size:
            features.append(0)
        
        return features
        
    def _has_strength_synergy(self, card, deck):
        """Check if card has strength synergy with deck"""
        strength_cards = ["demon form", "inflame", "spot weakness", "flex", "limit break"]
        heavy_scaling = ["heavy blade", "sword boomerang", "twin strike", "pummel"]
        
        # Card is a strength card
        if any(x in card.name.lower() for x in strength_cards):
            return True
            
        # Card benefits from strength
        if any(x in card.name.lower() for x in heavy_scaling):
            return True
            
        # Deck has strength cards
        return any(any(x in c.name.lower() for x in strength_cards) for c in deck)
    
    def _has_block_synergy(self, card, deck):
        """Check if card has block synergy with deck"""
        block_cards = ["barricade", "body slam", "entrench", "juggernaut", "metallicize"]
        return (any(x in card.name.lower() for x in block_cards) or
                any(any(x in c.name.lower() for x in block_cards) for c in deck))
    
    def _has_exhaust_synergy(self, card, deck):
        """Check if card has exhaust synergy with deck"""
        exhaust_synergy = ["feel no pain", "dark embrace", "sentinel", "corruption"]
        return (card.exhausts or
                any(x in card.name.lower() for x in exhaust_synergy) or
                any(any(x in c.name.lower() for x in exhaust_synergy) for c in deck))
    
    def _has_strike_synergy(self, card, deck):
        """Check if card has strike synergy"""
        return ("strike" in card.name.lower() or
                any("perfected strike" in c.name.lower() for c in deck))
    
    def _has_self_damage_synergy(self, card, deck):
        """Check if card has self-damage synergy"""
        self_damage_cards = ["rupture", "brutality", "combust", "offering"]
        self_damage_payoff = ["rupture", "limit break", "demon form"]
        
        # Card deals self damage
        if any(x in card.name.lower() for x in self_damage_cards):
            return True
            
        # Card benefits from self damage
        if any(x in card.name.lower() for x in self_damage_payoff):
            return True
            
        # Deck has self damage synergy
        return any(any(x in c.name.lower() for x in self_damage_cards + self_damage_payoff) for c in deck)
        

    def handle_hand_select(self, game_state: Game):
        """Neural network based hand card selection"""
        if not game_state.choice_available:
            return ProceedAction()
                
        cards = game_state.hand
        if not cards:
            return ProceedAction()
                
        # Get neural network evaluation of each card
        values = self.evaluate_cards(cards, game_state)
        
        # Select number of cards based on what the screen requires
        num_cards = min(game_state.screen.num_cards, len(cards))
        
        # For discarding, we want the lowest valued cards
        if "discard" in str(game_state.screen.screen_type).lower():
            _, indices = torch.topk(values, num_cards, largest=False)  # Get lowest valued cards
        else:
            _, indices = torch.topk(values, num_cards, largest=True)  # Get highest valued cards
            
        # Convert indices to list and ensure they're in valid range
        index_list = indices.tolist()
        if isinstance(index_list, int):
            index_list = [index_list]
        
        # Safety check to ensure indices are within bounds
        valid_indices = [i for i in index_list if i < len(cards)]
        if not valid_indices:
            valid_indices = [0]  # Default to first card if no valid indices
            
        # Select the cards
        selected_cards = [cards[i] for i in valid_indices]
        
        return CardSelectAction(selected_cards)

    def evaluate_shop_item(self, item, game_state: Game):
        """Evaluate shop items (cards/relics) for purchase"""
        item_features = []
        
        if hasattr(item, 'type'):  # Card
            item_features = [
                item.cost / game_state.gold,  # Cost relative to available gold
                1.0 if item.type == "ATTACK" else 0.0,
                1.0 if item.type == "SKILL" else 0.0,
                1.0 if item.type == "POWER" else 0.0,
                1.0 if item.exhausts else 0.0,
                item.damage / 50 if hasattr(item, 'damage') else 0,
                item.block / 30 if hasattr(item, 'block') else 0,
                1.0 if item.rarity == "RARE" else 0.5 if item.rarity == "UNCOMMON" else 0.0,
            ]
        else:  # Relic
            item_features = [
                item.price / game_state.gold,  # Cost relative to available gold
                1.0,  # Relic indicator
                0.0, 0.0, 0.0,  # Padding for card types
                0.0, 0.0,  # Padding for damage/block
            ]
            
        # Pad features
        while len(item_features) < self.input_size:
            item_features.append(0)
            
        # Get neural network evaluation
        state_tensor = torch.FloatTensor(item_features).unsqueeze(0)
        with torch.no_grad():
            value, _ = self.policy_net(state_tensor)  # Now returns tuple of (q_values, hidden)
            # Take mean of q_values as overall item value
            return value.mean().item()  # Convert to scalar properly

    def handle_shop(self, game_state: Game):
        """Neural network based shop decisions"""
        # Evaluate all available items
        available_items = []
        
        # Evaluate relics
        for relic in game_state.screen.relics:
            if game_state.gold >= relic.price:
                value = self.evaluate_shop_item(relic, game_state)
                available_items.append((value, "relic", relic))
                
        # Evaluate cards
        for card in game_state.screen.cards:
            if game_state.gold >= card.price:
                try:
                    value = self.evaluate_shop_item(card, game_state)
                    available_items.append((value, "card", card))
                except Exception as e:
                    print(f"Error evaluating card {card.name}: {e}")
                    continue
                
        if available_items:
            # Sort by value and check if best item is worth buying
            available_items.sort(reverse=True, key=lambda x: x[0])
            best_value, item_type, item = available_items[0]
            
            if best_value > 0.3:  # Threshold for purchase
                if item_type == "relic":
                    return BuyRelicAction(item)
                else:
                    return BuyCardAction(item)
                
        if game_state.screen.purge_available and game_state.gold >= game_state.screen.purge_cost:
            return ChooseAction(name="purge")
                    
        # If nothing worth buying or can't afford anything
        if game_state.cancel_available:
            return CancelAction()
        return ProceedAction()

    def evaluate_monster_threat(self, monster, game_state: Game):
        """Evaluate how threatening a monster is"""
        monster_features = [
            monster.current_hp / monster.max_hp,
            monster.block / 999,
            monster.move_adjusted_damage / 50 if monster.move_adjusted_damage else 0,
            monster.move_hits / 5 if monster.move_hits else 0,
            1.0 if monster.intent == Intent.ATTACK else 0.0,
            1.0 if monster.intent == Intent.BUFF else 0.0,
            1.0 if monster.intent == Intent.DEBUFF else 0.0,
        ]
        
        # Add game state context
        monster_features.extend([
            game_state.current_hp / game_state.max_hp,
            game_state.player.block / 200,
            game_state.act / 3,
        ])
        
        # Pad features
        while len(monster_features) < self.input_size:
            monster_features.append(0)
            
        # Get neural network evaluation
        state_tensor = torch.FloatTensor(monster_features).unsqueeze(0).unsqueeze(0).to(next(self.policy_net.parameters()).device)
        with torch.no_grad():
            threat_values, _ = self.policy_net(state_tensor)  # Unpack tuple
            # Take the mean of the output values as the threat score
            threat = torch.mean(threat_values).item()
                
        return threat

    def select_monster_target(self, game_state: Game):
        """Choose monster target using neural network"""
        available_monsters = [m for m in game_state.monsters 
                            if m.current_hp > 0 and not m.half_dead and not m.is_gone]
                            
        if not available_monsters:
            return None
            
        # Get threat assessment for each monster
        monster_threats = []
        for monster in available_monsters:
            threat = self.evaluate_monster_threat(monster, game_state)
            
            # Modify threat based on HP
            hp_factor = 1.5 if monster.current_hp / monster.max_hp < 0.5 else 1.0
            final_threat = threat * hp_factor
            
            # Store tuple of (threat_value, monster)
            monster_threats.append((final_threat, monster))
            
        # Sort by the threat value (first element of tuple)
        monster_threats.sort(key=lambda x: x[0], reverse=True)
        
        # Return the monster with highest threat
        return monster_threats[0][1] if monster_threats else None

    def handle_combat_reward(self, game_state: Game):
        """Handle combat reward screens"""
        # Take all non-card rewards first
        for reward_item in game_state.screen.rewards:
            return CombatRewardAction(reward_item)
                
        # After taking other rewards, look at card rewards
        return ProceedAction()
    
    def handle_card_reward(self, game_state: Game):
        """Handle card reward selection screen"""
        logger = GameStateLogger()
        logger.log_error("ATTEMPTING TO PICK CARD")

        # Evaluate all available cards
        card_values = self.evaluate_cards(game_state.screen.cards, game_state)
        
        # Make sure the values tensor is the same length as the cards list
        assert len(card_values) == len(game_state.screen.cards), f"Mismatch between values ({len(card_values)}) and cards ({len(game_state.screen.cards)})"
        
        best_card_idx = card_values.argmax().item()
        best_card_value = card_values[best_card_idx].item()
        
        logger.log_error(f"Number of cards: {len(game_state.screen.cards)}")
        logger.log_error(f"Best card index: {best_card_idx}")
        logger.log_error(f"Best card value: {best_card_value}")
        logger.log_error(f"Card choices: {[card.name for card in game_state.screen.cards]}")
        
        # If the best card is above our threshold, take it
        if best_card_value > 0.002 or not game_state.cancel_available:  # Threshold for taking a card
            logger.log_error(f"Selecting card: {game_state.screen.cards[best_card_idx].name}")
            return ChooseAction(choice_index=best_card_idx)
        
        # Skip if possible, otherwise proceed
        logger.log_error("Skipping card selection")
        return CancelAction()

    def handle_event_choice(self, game_state: Game):
        """Handle event choices using neural network"""
        logger = GameStateLogger()
        logger.log_error("=== DECISION STATE ===")
        
        # First verify we have valid choices
        if not hasattr(game_state, 'choice_list') or not game_state.choice_list:
            logger.log_error("No choices available in event")
            return ProceedAction()
            
        # Convert event state to features
        event_features = [
            game_state.current_hp / game_state.max_hp,
            game_state.gold / 999,
            game_state.act / 3,
            len(game_state.deck) / 40,  # Normalized deck size
            len(game_state.relics) / 15,  # Normalized relic count
        ]
        
        try:
            # Add features for each choice in choice_list
            for choice in game_state.choice_list:
                choice_text = choice.lower()
                choice_features = [
                    1.0 if "gold" in choice_text else 0.0,
                    1.0 if ("lose" in choice_text and "hp" in choice_text) else 0.0,
                    1.0 if "card" in choice_text else 0.0,
                    1.0 if "relic" in choice_text else 0.0,
                ]
                event_features.extend(choice_features)
            
            # Pad features
            while len(event_features) < self.input_size:
                event_features.append(0)
                
            # Get neural network evaluation
            state_tensor = torch.FloatTensor(event_features).unsqueeze(0)
            with torch.no_grad():
                values = self.policy_net(state_tensor)
                
            # Choose best option
            best_choice = values.argmax().item()
            
            # Verify the choice index is valid
            if best_choice >= len(game_state.choice_list):
                logger.log_error(f"Invalid choice index {best_choice}, defaulting to 0")
                best_choice = 0
                
            return ChooseAction(choice_index=best_choice)
            
        except Exception as e:
            logger.log_error(f"Error in handle_event_choice: {str(e)}")
            # Default to first choice if there's an error
            return ChooseAction(choice_index=0)
        
    def evaluate_potion_use(self, potion, game_state):
        """Evaluate whether to use a potion using the neural network"""
        from neuralAgent import NeuralAgent
        
        if potion is None or not hasattr(potion, 'name'):
            return 0.0
        
        if 'colorless' in potion.name:
            return 0.0
        
        # Get base state features
        state_tensor = NeuralAgent().state_to_tensor(game_state, True)  # This returns [1, 1, state_size]
        
        # Get potion features and match dimensions
        potion_features = torch.tensor(self.encode_potion_features(potion), 
                                    dtype=torch.float32,
                                    device=state_tensor.device)
        # Reshape potion features to match state tensor dimensions
        potion_features = potion_features.unsqueeze(0).unsqueeze(0)  # Now [1, 1, potion_feature_size]
        
        # Combine features along the last dimension
        combined_features = torch.cat([state_tensor, potion_features], dim=-1)
        
        # Get neural network evaluation
        with torch.no_grad():
            value, _ = self.policy_net(combined_features)
            return value.mean().item()
          
    
    def encode_potion_features(self, potion):
        """Encode potion features based on name and properties"""
        if not potion:
            return [0] * 8  # Return zero vector for empty potion slot
        
        features = []
        name_lower = potion.name.lower()
        
        # Basic properties (3 features)
        features.extend([
            1 if potion.can_use else 0,
            1 if potion.requires_target else 0,
            1 if potion.can_discard else 0,
        ])
        
        # Potion type classification (5 features)
        features.extend([
            # Offensive potions
            1 if any(x in name_lower for x in ["fire", "explosive", "poison", "attack"]) else 0,
            
            # Defensive potions
            1 if any(x in name_lower for x in ["block", "ghost", "fairy", "heart", "steel"]) else 0,
            
            # Buff potions
            1 if any(x in name_lower for x in ["strength", "dexterity", "speed", "energy"]) else 0,
            
            # Debuff potions
            1 if any(x in name_lower for x in ["weak", "fear", "vulnerable"]) else 0,
            
            # Utility potions
            1 if any(x in name_lower for x in ["swift", "cultist", "power", "skill"]) else 0,
        ])
        
        return features

    def encode_potions_state(self, game_state: Game):
        """Encode all potion slots"""
        potion_features = []
        
        # Encode each potion slot (max 3 slots)
        for potion in game_state.potions[:3]:
            potion_features.extend(self.encode_potion_features(potion))
        
        # Pad if less than 3 potions
        while len(potion_features) < 24:  # 8 features * 3 slots
            potion_features.extend([0] * 8)
        
        # Add general potion state
        potion_features.extend([
            len([p for p in game_state.potions if p]) / 3,  # Potion slot utilization
            1 if len([p for p in game_state.potions if p]) == 3 else 0,  # Full potion slots
        ])
        
        return potion_features