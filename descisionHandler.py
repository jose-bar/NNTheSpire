from spirecomm.spire.game import Game
from spirecomm.communication.action import *
from spirecomm.spire.character import Intent
from logger import GameStateLogger
import torch
import torch.nn.functional as F

class DecisionHandler:
    def __init__(self, policy_net, input_size=256):
        self.policy_net = policy_net
        self.input_size = input_size
        
    def evaluate_cards(self, cards, game_state: Game):
        """Evaluate a list of cards using the neural network"""
        card_features = []
        for card in cards:
            # Convert card to feature vector
            features = [
                card.cost / 3,  # Normalized cost
                1.0 if card.type == "ATTACK" else 0.0,
                1.0 if card.type == "SKILL" else 0.0,
                1.0 if card.type == "POWER" else 0.0,
                1.0 if card.exhausts else 0.0,
                card.damage / 50 if hasattr(card, 'damage') else 0,
                card.block / 30 if hasattr(card, 'block') else 0,
                1.0 if card.rarity == "RARE" else 0.5 if card.rarity == "UNCOMMON" else 0.0,
            ]
            card_features.extend(features)
            
        # Pad features to fixed size
        while len(card_features) < self.input_size:
            card_features.append(0)
            
        # Get neural network evaluation
        state_tensor = torch.FloatTensor(card_features).unsqueeze(0)
        with torch.no_grad():
            values = self.policy_net(state_tensor)
        return values.squeeze()

    def handle_hand_select(self, game_state: Game):
        """Neural network based hand card selection"""
        if not game_state.choice_available:
            return ProceedAction()
            
        cards = game_state.hand
        if not cards:
            return ProceedAction()
            
        # Get neural network evaluation of each card
        values = self.evaluate_cards(cards, game_state)
        
        # Select cards based on required number
        num_cards = min(game_state.screen.num_cards, len(cards))
        _, indices = torch.topk(values, num_cards)
        selected_cards = [cards[i] for i in indices]
        
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
                1.0 if "boss" in item.id.lower() else 0.5,  # Relic rarity approximation
            ]
            
        # Pad features
        while len(item_features) < self.input_size:
            item_features.append(0)
            
        # Get neural network evaluation
        state_tensor = torch.FloatTensor(item_features).unsqueeze(0)
        with torch.no_grad():
            value = self.policy_net(state_tensor)
        return value.item()

    def handle_shop(self, game_state: Game):
        """Neural network based shop decisions"""
        if game_state.screen.purge_available and game_state.gold >= game_state.screen.purge_cost:
            return ChooseAction(name="purge")
            
        # Evaluate all available items
        available_items = []
        for relic in game_state.screen.relics:
            if game_state.gold >= relic.price:
                value = self.evaluate_shop_item(relic, game_state)
                available_items.append((value, "relic", relic))
                
        for card in game_state.screen.cards:
            if game_state.gold >= card.price:
                value = self.evaluate_shop_item(card, game_state)
                available_items.append((value, "card", card))
                
        if available_items:
            # Sort by value and check if best item is worth buying
            available_items.sort(reverse=True)
            best_value, item_type, item = available_items[0]
            
            if best_value > 0.5:  # Threshold for purchase
                if item_type == "relic":
                    return BuyRelicAction(item)
                else:
                    return BuyCardAction(item)
                    
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
            game_state.player.block / 999,
            game_state.act / 3,
        ])
        
        # Pad features
        while len(monster_features) < self.input_size:
            monster_features.append(0)
            
        # Get neural network evaluation
        state_tensor = torch.FloatTensor(monster_features).unsqueeze(0)
        with torch.no_grad():
            threat = self.policy_net(state_tensor)
        return threat.item()

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
            hp_factor = 1.5 if monster.current_hp / monster.max_hp < 0.3 else 1.0
            threat *= hp_factor
            
            monster_threats.append((threat, monster))
            
        # Select highest threat monster
        monster_threats.sort(reverse=True)
        return monster_threats[0][1]

    def evaluate_combat_reward(self, reward_item, game_state: Game):
        """Evaluate combat reward items"""
        # Always take gold, potions (if space), and relics
        if reward_item.reward_type in [RewardType.GOLD, RewardType.RELIC]:
            return 1.0
        if reward_item.reward_type == RewardType.POTION and not game_state.are_potions_full():
            return 1.0
            
        if reward_item.reward_type == RewardType.CARD:
            card_value = self.evaluate_cards([reward_item.card], game_state)
            return card_value.item()
            
        return 0.0

    def handle_event_choice(self, game_state: Game):
        """Handle event choices using neural network"""
        # Convert event state to features
        event_features = [
            game_state.current_hp / game_state.max_hp,
            game_state.gold / 999,
            game_state.act / 3,
            len(game_state.deck) / 40,  # Normalized deck size
            len(game_state.relics) / 15,  # Normalized relic count
        ]
        
        # Add features for each choice
        choices = game_state.screen.options
        for choice in choices:
            # Extract keywords from choice text
            has_gold = "gold" in choice.text.lower()
            has_hp_loss = "lose" in choice.text.lower() and "hp" in choice.text.lower()
            has_card = "card" in choice.text.lower()
            has_relic = "relic" in choice.text.lower()
            
            choice_features = [
                1.0 if has_gold else 0.0,
                1.0 if has_hp_loss else 0.0,
                1.0 if has_card else 0.0,
                1.0 if has_relic else 0.0,
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
        return ChooseAction(choice_index=best_choice)