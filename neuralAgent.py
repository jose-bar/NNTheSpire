from main import *
from logger import GameStateLogger
from reward_handler import RewardHandler


class NeuralAgent:
    def __init__(self):
        # For tracking in ghame things
        self.visited_shop = False
        self.logger = GameStateLogger()
        self.reward_handler = RewardHandler()
        
        self.input_size = 256  # Adjust based on your state representation
        self.hidden_size = 128
        self.output_size = 64  # Adjust based on possible actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = SlayerNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.target_net = SlayerNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
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
        
    def optimize_model(self):
        """Perform one step of optimization on the policy network"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample random transitions
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.cat([s.unsqueeze(0) for s in batch[0]])
        action_batch = torch.tensor([a for a in batch[1]], device=self.device)
        reward_batch = torch.tensor([r for r in batch[2]], device=self.device)
        next_state_batch = torch.cat([s.unsqueeze(0) for s in batch[3]])
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        
        # Compute expected Q values
        expected_q_values = reward_batch + (self.gamma * next_q_values)
        
        # Compute loss and optimize
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        if self.games_played % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def state_to_tensor(self, game_state: Game):
        """Convert game state to tensor format"""
        state = []
        
        # Basic game state information
        state.extend([
            game_state.current_hp / game_state.max_hp,
            game_state.gold / 999,  # Normalized gold
            game_state.floor / 20,   # Normalized floor
        ])
        
        # Combat state
        if game_state.in_combat:
            # Add player state
            state.extend([
                game_state.player.block / 999,
                game_state.player.energy / 999,
            ])
            
            # Add monster states (pad if necessary)
            monster_features = []
            for monster in game_state.monsters[:5]:  # Consider up to 3 monsters
                monster_features.extend([
                    monster.current_hp / monster.max_hp,
                    monster.block / 999,
                    monster.intent.value / 17,  # Normalized intent value
                ])
            
            # Pad if less than 3 monsters
            while len(monster_features) < 15:  # 5 monsters * 3 features
                monster_features.extend([0, 0, 0, 0, 0])
            
            state.extend(monster_features)
        else:
            state.extend([0] * 11)  # Pad with zeros when not in combat
            
        # Pad to fixed size
        while len(state) < self.input_size:
            state.append(0)
            
        return torch.FloatTensor(state).to(self.device)
    

    def get_next_action_in_game(self, game_state: Game):
        """Choose next action during gameplay with improved error handling"""
        reward = self.reward_handler.calculate_reward(game_state)
        current_state = self.state_to_tensor(game_state)
        
        # Store previous state transition
        if len(self.memory) > 0:
            prev_state, prev_action, _, _ = self.memory[-1]
            self.memory[-1] = (prev_state, prev_action, reward, current_state)

        if game_state.screen_type == ScreenType.GAME_OVER:
            # Record game result before starting new game
            # Return to menu and start new game
            # return self.handle_game_over(game_state)
            self.logger.log_game_state(game_state)
            return ProceedAction()

        # Handle card select screens (for discard, exhaust, etc.)
        elif game_state.screen_type == ScreenType.HAND_SELECT:
            if not game_state.choice_available:
                self.logger.log_game_state(game_state)
                return ProceedAction()
            num_cards = min(game_state.screen.num_cards, len(game_state.hand))
            cards_to_select = game_state.hand[:num_cards]  # Select first N cards
            return CardSelectAction(cards_to_select)

        # Enhanced shop screen handling
        elif game_state.screen_type == ScreenType.SHOP_ROOM:
            if not self.visited_shop:
                self.visited_shop = True
                return ChooseAction(name="shop")
            else:
                self.visited_shop = False
                return ProceedAction()

        elif game_state.screen_type == ScreenType.SHOP_SCREEN:
            # First try to make purchases
            if game_state.screen.purge_available and game_state.gold >= game_state.screen.purge_cost:
                return ChooseAction(name="purge")
            for relic in game_state.screen.relics:
                if game_state.gold >= relic.price:
                    return BuyRelicAction(relic)
            for card in game_state.screen.cards:
                # Ball out
                if game_state.gold >= card.price:
                    return BuyCardAction(card)
            
            # If we can't buy anything or done shopping, leave
            if game_state.cancel_available:
                return CancelAction()
            
            # If still in some shop sub-screen
            if game_state.proceed_available:
                return ProceedAction()
            self.logger.log_game_state(game_state)
            return StateAction()

        # Rest site handling
        elif game_state.screen_type == ScreenType.REST:
            return self.choose_rest_option(game_state)

        # Combat logic with unplayable card handling
        elif game_state.in_combat:
            if game_state.play_available and len(game_state.hand) > 0:
                playable_cards = [card for card in game_state.hand if card.is_playable]
                if playable_cards:  # Only try to play if we have playable cards
                    if random.random() < self.epsilon:
                        card = random.choice(playable_cards)
                    else:
                        # Use neural network
                        with torch.no_grad():
                            action_values = self.policy_net(current_state)
                            action_idx = action_values.argmax().item()
                            if action_idx < len(playable_cards):
                                card = playable_cards[action_idx]
                            else:
                                card = playable_cards[0]
                    
                    # Handle targeting
                    if card.has_target and game_state.monsters:
                        available_monsters = [m for m in game_state.monsters 
                                        if m.current_hp > 0 and not m.half_dead and not m.is_gone]
                        if available_monsters:
                            target = min(available_monsters, key=lambda x: x.current_hp)
                            return PlayCardAction(card=card, target_monster=target)
                        return EndTurnAction()
                    return PlayCardAction(card=card)
                elif game_state.end_available:
                    return EndTurnAction()
                
            if game_state.end_available:
                return EndTurnAction()
            
            self.logger.log_game_state(game_state)
            return StateAction()

        elif game_state.screen_type == ScreenType.COMBAT_REWARD:
            # Skip potion rewards if potion slots are full
            for reward_item in game_state.screen.rewards:
                # Skip potions if slots are full
                if reward_item.reward_type == RewardType.POTION and game_state.are_potions_full():
                    continue
                else:
                    return CombatRewardAction(reward_item)
                    
            # If we can't take any rewards, proceed
            self.skipped_cards = False  # Reset skipped cards flag
            return ProceedAction()
    
        # Generic screen handling
        else:
            if game_state.choice_available:
                return ChooseAction(choice_index=0)
            elif game_state.proceed_available:
                return ProceedAction()
            elif game_state.cancel_available:
                return CancelAction()
            return StateAction()

        # Store current state-action for next iteration
        # self.memory.append((current_state, None, None, None))  # action_idx will be set next iteration
        
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
            return ProceedAction()
         
        
    
    def get_next_action_out_of_game(self, game_state=None):
        """Choose next action when out of combat or in game setup"""
        # If we have a current class, start the game with it
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
        self.logger.log_game_state(game_state)
        return StateAction()
    
    def handle_error(self, error_msg):
        """Handle error messages from the game"""
        print(f"Error: {error_msg}")
        
        return ProceedAction()
    
    def change_class(self, new_class):
        """Change the current character class"""
        self.current_class = new_class
        
        
    def handle_grid_screen(self, game_state: Game):
        """Handle GRID type screens (card selection, upgrades, etc)"""
        self.logger.log_grid_screen(game_state)
        
        # If we're confirming a selection
        if game_state.screen.confirm_up:
            return ChooseAction(name="confirm")
        
        # Get available cards and their context
        grid_cards = game_state.screen.cards
        if not grid_cards:  # No cards to select
            if game_state.cancel_available:
                return CancelAction()
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
            
        elif "headbutt" in game_state.screen.last_card_played.lower() or "hologram" in game_state.screen.last_card_played.lower() or "exhume" in game_state.screen.last_card_played.lower(): 
            # Headbutt card selection (from discard pile)
            return self._select_card_for_return(grid_cards)
            
        else:
            # Default card selection strategy
            return self._select_default_card(grid_cards)

    def _select_card_for_upgrade(self, cards):
        """Prioritize cards for upgrade"""
        # Priority order: Attacks > Skills > Powers > Status/Curse
        priority_order = {
            "ATTACK": 1,
            "SKILL": 2,
            "POWER": 3,
            "STATUS": 0,
            "CURSE": 0
        }
        
        # Sort cards by priority
        best_card = max(cards, key=lambda x: (
            priority_order.get(x.type, 0),  # Card type priority
            -x.cost,  # Higher cost cards first
            x.rarity in ["RARE", "UNCOMMON"]  # Rarer cards preferred
        ))
        
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