from main import *
from logger import GameStateLogger
from reward_handler import RewardHandler
from descisionHandler import DecisionHandler

class NeuralAgent:
    def __init__(self):
        # For tracking in ghame things
        self.visited_shop = False
        self.checked_rewards = False
        self.seen_card_reward = False
        self.logger = GameStateLogger()
        self.reward_handler = RewardHandler()
        
        self.input_size = 32  # Adjust based on your state representation
        self.hidden_size = 32
        self.output_size = 32  # Adjust based on possible actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = SlayerNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(self.device)
        
        self.target_net = SlayerNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(self.device)
        
        self.decision_handler = DecisionHandler(self.policy_net, self.input_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize hidden states
        self.hidden_state = None
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        
        self.current_class = None
        self.epsilon = 0.9  # Exploration rate
        
        # State tracking for reward calculation
        self.previous_hp = None
        self.combat_starting_hp = None
        self.in_combat_previously = False
        self.total_damage_taken = 0
        self.victories = 0
        self.games_played = 0
        
        self.memory = deque(maxlen=10000)
        self.action_history = []  # Track actions for current episode
        self.state_history = []   # Track states for current episode
        self.reward_history = []  # Track rewards for current episode
        
    @staticmethod
    def encode_game_state(game_state, skip):
        """Optimized game state encoding with reduced dimensions"""
        state = []
        
        # Core player state (5 features)
        if skip:
            state.extend([
                game_state.current_hp / game_state.max_hp,  # HP percentage
                game_state.gold / 999,  # Normalized gold
                game_state.act / 3,  # Current act
                len(game_state.deck) / 40,  # Normalized deck size
            ])
        else:
            state.extend([
                game_state.current_hp / game_state.max_hp,  # HP percentage
                game_state.gold / 999,  # Normalized gold
                game_state.act / 3,  # Current act
                len(game_state.deck) / 40,  # Normalized deck size
                1 if game_state.in_combat else 0  # Combat flag
            ])
        
        # Combat-specific features (8 features)
        if game_state.in_combat:
            hand = game_state.hand
            state.extend([
                game_state.player.block / 50,  # Current block
                game_state.player.energy / 3,  # Available energy
                len([c for c in hand if c.type == "ATTACK"]) / max(1, len(hand)),  # Attack ratio
                len([c for c in hand if c.type == "SKILL"]) / max(1, len(hand)),   # Skill ratio
                len([c for c in hand if c.type == "POWER"]) / max(1, len(hand)),   # Power ratio
                len([c for c in hand if c.is_playable]) / max(1, len(hand)),       # Playable ratio
                len([c for c in hand if c.cost == 0]) / max(1, len(hand)),         # Zero cost ratio
                game_state.turn_count / 10 if hasattr(game_state, 'turn_count') else 0  # Turn number
            ])
        else:
            state.extend([0] * 8)
        
        # Monster state (12 features - 3 monsters x 4 features each)
        max_monsters = 3
        monster_features = []
        
        for monster in game_state.monsters[:max_monsters]:
            if monster and not monster.is_gone:
                monster_features.extend([
                    monster.current_hp / max(1, monster.max_hp),  # HP ratio
                    monster.move_adjusted_damage / 40 if monster.move_adjusted_damage else 0,  # Damage
                    1 if monster.intent and "ATTACK" in str(monster.intent) else 0,  # Attack intent
                    1 if monster.intent and ("BUFF" in str(monster.intent) or "DEBUFF" in str(monster.intent)) else 0  # Status intent
                ])
            else:
                monster_features.extend([0] * 4)
        
        # Pad if we have fewer than max_monsters
        remaining_monsters = max_monsters - len(game_state.monsters)
        if remaining_monsters > 0:
            monster_features.extend([0] * (remaining_monsters * 4))
        
        state.extend(monster_features)
        
        if not skip:
            # Screen state (7 features)
            state.extend([
                1 if game_state.screen_type == ScreenType.REST else 0,
                1 if game_state.screen_type == ScreenType.SHOP_ROOM else 0,
                1 if game_state.screen_type == ScreenType.EVENT else 0,
                1 if game_state.screen_type == ScreenType.COMBAT_REWARD else 0,
                1 if game_state.screen_type == ScreenType.CARD_REWARD else 0,
                1 if game_state.proceed_available else 0,
                1 if game_state.choice_available else 0
            ])
        
        # Total features: 32 (5 + 8 + 12 + 7)
        if not skip:
            assert len(state) == 32, f"State length is {len(state)}, expected 32"
        else:
            assert len(state) == 24
        
        return torch.FloatTensor(state)
    
    def state_to_tensor(self, game_state: Game, skip):
        """Convert game state to tensor format using our new encoding"""
        # Get the base encoded state
        state = self.encode_game_state(game_state, skip)
        
        # Add batch and sequence dimensions: [64] -> [1, 1, 64]
        state = state.unsqueeze(0).unsqueeze(0)
        
        return state.to(self.device)

    def store_transition(self, state, action, reward, next_state):
        """Store a transition in replay memory"""
        # Convert states to tensors if they aren't already
        if not isinstance(state, torch.Tensor):
            state = self.state_to_tensor(state, False)
        if not isinstance(next_state, torch.Tensor):
            next_state = self.state_to_tensor(next_state, False)
            
        self.memory.append((state, action, reward, next_state))
        
    def optimize_model(self):
        """Train the neural network using stored transitions"""
        if len(self.memory) < self.batch_size:
            return
            
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        # States are already tensors from store_transition
        state_batch = torch.cat([s.unsqueeze(0) for s in batch[0]])
        action_batch = torch.cat([a for a in batch[1]])
        reward_batch = torch.tensor([r for r in batch[2]], device=self.device)
        next_state_batch = torch.cat([s.unsqueeze(0) for s in batch[3]])
        
        # Get current Q values with hidden states
        current_q_values, _ = self.policy_net(state_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1))
        
        # Get next Q values with target network
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_state_batch)
            next_q_values = next_q_values.max(1)[0]
            expected_q_values = reward_batch + (self.gamma * next_q_values)
        
        # Compute loss and optimize
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Periodically update target network
        if self.games_played % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            

    def get_next_action_in_game(self, game_state: Game):
        
        self.logger.log_game_state(game_state)
        
        """Choose next action during gameplay with improved error handling"""
        reward = self.reward_handler.calculate_reward(game_state)
        current_state = self.state_to_tensor(game_state, False)
        
        # Store previous state transition
        if len(self.memory) > 0:
            prev_state, prev_action, _, _ = self.memory[-1]
            self.store_transition(prev_state, prev_action, reward, current_state)
        
        with torch.no_grad():
            q_values, self.hidden_state = self.policy_net(
                current_state,
                self.hidden_state
            )
        
        if game_state.screen_type == ScreenType.GAME_OVER:
            # Return to menu and start new game
            self.logger.log_game_state(game_state)
            return ProceedAction()
        

        # Handle card select screens (for discard, exhaust, etc.)
        elif game_state.screen_type == ScreenType.HAND_SELECT:
            return self.decision_handler.handle_hand_select(game_state)

        # Enhanced shop screen handling
        elif game_state.screen_type == ScreenType.SHOP_ROOM:
            if not self.visited_shop:
                self.visited_shop = True
                return ChooseAction(name="shop")
            else:
                self.visited_shop = False
                return ProceedAction()

        elif game_state.screen_type == ScreenType.SHOP_SCREEN:
            return self.decision_handler.handle_shop(game_state)

        # Rest site handling
        elif game_state.screen_type == ScreenType.REST:
            return self.choose_rest_option(game_state)

        elif game_state.screen_type == ScreenType.COMBAT_REWARD:
            if not self.checked_rewards:
                for reward_item in game_state.screen.rewards:
                    if reward_item.reward_type == RewardType.CARD and not self.seen_card_reward:
                        self.seen_card_reward = True
                        self.checked_rewards = True
                        return CombatRewardAction(reward_item)
                    # Handle potions - only take if we have space
                    elif reward_item.reward_type == RewardType.POTION:
                        if not game_state.are_potions_full():
                            return CombatRewardAction(reward_item)
                        else:
                            continue  # Skip this potion if slots are full
                    # Handle other rewards (gold, relics, etc)
                    elif reward_item.reward_type != RewardType.CARD:
                        return CombatRewardAction(reward_item)
                self.checked_rewards = True
                return ProceedAction()
            else:
                self.checked_rewards = False
                return ProceedAction()
        
        elif game_state.screen_type == ScreenType.CARD_REWARD:
            if not game_state.in_combat:
                if self.seen_card_reward:
                    self.seen_card_reward = False  # Reset for next combat
                    return self.decision_handler.handle_card_reward(game_state)
                return CancelAction()
            else:
                return self.decision_handler.handle_card_reward(game_state)

        elif game_state.screen_type == ScreenType.EVENT:
            #theres only one fucking choice
            if(len(game_state.screen.options)) == 1:
                return ChooseAction(0)
            return self.decision_handler.handle_event_choice(game_state)

        # Combat logic with unplayable card handling
        elif game_state.in_combat and game_state.screen_type is not ScreenType.GRID:
            potions = game_state.get_real_potions()
            for potion in potions:
                if potion.can_use:
                    use_potion = self.decision_handler.evaluate_potion_use(potion, game_state)
                    if use_potion > 0.005:  # Threshold for potion use
                        if potion.requires_target:
                            target = self.decision_handler.select_monster_target(game_state)
                            if target:
                                return PotionAction(True, potion=potion, target_monster=target)
                        else:
                            return PotionAction(True, potion=potion)
            
            if game_state.play_available and len(game_state.hand) > 0:
                playable_cards = [card for card in game_state.hand if card.is_playable]
                if playable_cards:  # Only try to play if we have playable cards
                    if random.random() < self.epsilon:
                        card = random.choice(playable_cards)
                    else:
                        # Use q_values to select action
                        action_idx = q_values.squeeze().argmax().item()  # Remove extra dimensions
                        if action_idx < len(playable_cards):
                            card = playable_cards[action_idx]
                        else:
                            card = playable_cards[0]
                    
                    # Handle targeting
                    if card.has_target and game_state.monsters:
                        target = self.decision_handler.select_monster_target(game_state)
                        if target:
                            return PlayCardAction(card=card, target_monster=target)
                    return PlayCardAction(card=card)
                
            if game_state.end_available:
                return EndTurnAction()
            
            self.logger.log_game_state(game_state)
            return StateAction()
                    
            # If we can't take any rewards, proceed
            self.skipped_cards = False  # Reset skipped cards flag
            return ProceedAction()
        
        elif game_state.screen_type == ScreenType.GRID:
            return self.handle_grid_screen(game_state)
        
        # Generic screen handling
        else:
            if game_state.choice_available:
                return ChooseAction(choice_index=0)
            elif game_state.proceed_available:
                self.logger.log_game_state(game_state)
                return ProceedAction()
            elif game_state.cancel_available:
                return CancelAction()
            return StateAction()
        
    
    def choose_rest_option(self, game_state: Game):
        """Enhanced rest site decision making"""
        rest_options = game_state.screen.rest_options
        
        if len(rest_options) > 0 and not game_state.screen.has_rested:
            # Critical HP threshold
            if RestOption.REST in rest_options and game_state.current_hp < game_state.max_hp / 2:
                return RestAction(RestOption.REST)
            
            # Pre-boss rest check (every 17th floor except act 1)
            elif (RestOption.REST in rest_options and 
                game_state.act != 1 and 
                game_state.floor % 17 == 15 and 
                game_state.current_hp < game_state.max_hp * 0.9):
                return RestAction(RestOption.REST)
                
            # Additional rest options in priority order
            elif RestOption.LIFT in rest_options:
                return RestAction(RestOption.LIFT)
                
            elif RestOption.DIG in rest_options:
                return RestAction(RestOption.DIG)
            
            # Prioritize upgrade if healthy
            elif RestOption.SMITH in rest_options:
                return RestAction(RestOption.SMITH)
                
            # Rest if any HP is missing
            elif RestOption.REST in rest_options and game_state.current_hp < game_state.max_hp:
                return RestAction(RestOption.REST)
                
            # Default choice if nothing else applies
            else:
                return ChooseAction(0)
        else:
            self.logger.log_game_state(game_state)
            return ProceedAction()
         
        
    
    def get_next_action_out_of_game(self, game_state=None):
        """Choose next action when out of combat or in game setup"""
        # If we have a current class, start the game with it
        
        self.logger.log_game_state(game_state)
        
        if self.current_class is not None:
            return StartGameAction(self.current_class)
        
        # If we have game state, check available commands
        if game_state is not None:
            if game_state.proceed_available:
                self.logger.log_game_state(game_state)
                return ProceedAction()
            elif game_state.choice_available:
                return ChooseAction(choice_index=0)
            elif game_state.cancel_available:
                return CancelAction()
            
        # If nothing else is available, request state
        return StateAction()
    
    def handle_error(self, error_msg):
        logger = GameStateLogger()
        """Handle error messages from the game"""
        self.logger.log_error("BRUHHHHHHH")
        return ProceedAction()
    
    def change_class(self, new_class):
        """Change the current character class"""
        self.current_class = new_class
        
        
    def handle_grid_screen(self, game_state: Game):
        """Handle GRID type screens (card selection, upgrades, etc)"""
        last_card_played = getattr(game_state.screen, 'last_card_played', None)
        
        # If we're confirming a selection
        if game_state.screen.confirm_up:
            return ChooseAction(name="confirm")
        
        # Get available cards and their context
        grid_cards = game_state.screen.cards
        if not grid_cards:  # No cards to select
            if game_state.cancel_available:
                return CancelAction()
            self.logger.log_game_state(game_state)
            return ProceedAction()
        
        # Different selection strategies based on context
        if game_state.screen.for_upgrade:
            # Prioritize key cards for upgrade
            return self._select_card_for_upgrade(grid_cards)
        
        elif game_state.screen.for_purge:
            # Select cards to remove from deck
            return self._select_card_for_purge(grid_cards)
            
        elif game_state.screen.for_transform:
            # Select cards to transform
            return self._select_card_for_transform(grid_cards)
            
        elif last_card_played and ("headbutt" in last_card_played.lower() or "hologram" in last_card_played.lower() or "exhume" in last_card_played.lower()):
                # Headbutt card selection (from discard pile)
                return self._select_card_for_return(grid_cards)
            
        else:
            # Default card selection strategy
            return self._select_default_card(grid_cards)

    def _select_card_for_upgrade(self, cards):
        """Prioritize cards for upgrade: Power > Skill > Attack, split by rarity, Bash over basic cards"""
        def get_card_priority(card):
            name_lower = card.name.lower()
            
            # Special case for Bash
            if name_lower == "bash":
                return (4, 3)  # Highest priority tier
            
            if card.rarity == 1:
                return (0, 0)
                
            # Type priority: Power (3) > Skill (2) > Attack (1)
            type_priority = {
                "POWER": 3,
                "SKILL": 2,
                "ATTACK": 1,
                "STATUS": 0,
                "CURSE": 0
            }
            
            # Rarity priority
            rarity_priority = {
                "RARE": 3,
                "UNCOMMON": 2,
                "COMMON": 1,
                "BASIC": -2
            }
            
            return (type_priority.get(str(card.type), 0),
                    rarity_priority.get(str(card.rarity), 0))
        
        best_card = max(cards, key=get_card_priority)
        return ChooseAction(name=best_card.name)

    def _select_card_for_purge(self, cards):
        """Select cards to remove from deck"""
        # Priority order for removal: Curse > Status > Basic Strikes/Defends > Others
        priority_order = {
            "CURSE": 5,
            "STATUS": 4,
            "Strike": 3,
            "Defend": 2,
            "BASIC": 1,
            "OTHER": 0
        }
        
        def get_card_priority(card):
            if card.type == "CURSE":
                return priority_order["CURSE"]
            elif card.type == "STATUS":
                return priority_order["STATUS"]
            elif "Strike" in card.name:
                return priority_order["Strike"]
            elif "Defend" in card.name:
                return priority_order["Defend"]
            elif card.rarity == "BASIC":
                return priority_order["BASIC"]
            return priority_order["OTHER"]
        
        best_card = max(cards, key=get_card_priority)
        return ChooseAction(name=best_card.name)

    def _select_card_for_transform(self, cards):
        """Select cards to transform"""
        # Similar to purge priority but slightly different
        # Prefer transforming basic cards as they're guaranteed to become something better
        return self._select_card_for_purge(cards)  # Using same logic for now

    def _select_card_for_return(self, cards):
        """Select card to return from discard pile"""
        # Priority: High impact cards, Powers, High cost cards
        best_card = max(cards, key=lambda x: (
            x.type == "POWER",  # Powers are good to get back
            x.cost,  # Higher cost cards usually have bigger effects
            x.rarity in ["RARE", "UNCOMMON"]  # Prefer rarer cards
        ))
        
        return ChooseAction(name=best_card.name)

    def _select_default_card(self, cards):
        """Default card selection strategy"""
        # Basic strategy: prefer rare cards, then uncommon, then common
        rarity_values = {
            "RARE": 3,
            "UNCOMMON": 2,
            "COMMON": 1,
            "BASIC": 0,
            "SPECIAL": 0,
            "CURSE": -1
        }
        
        best_card = max(cards, key=lambda x: (
            rarity_values.get(x.rarity, 0),
            -x.cost  # Lower cost preferred in general selection
        ))
        
        return ChooseAction(name=best_card.name)
    
