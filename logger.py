import logging
from spirecomm.spire.game import Game
from spirecomm.spire.screen import ScreenType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('slay_the_spire_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class GameStateLogger:
    @staticmethod
    def log_game_state(game_state: Game):
        """Log detailed information about the current game state"""
        logger.info("\n=== GAME STATE ===")
        logger.info(f"Screen Type: {game_state.screen_type}")
        # logger.info(f"Room Type: {game_state.room_type}")
        # logger.info(f"Floor: {game_state.floor}")
        # logger.info(f"Act: {game_state.act}")
        # logger.info(f"Current HP: {game_state.current_hp}/{game_state.max_hp}")
        # logger.info(f"Gold: {game_state.gold}")
        # logger.info(f"In Combat: {game_state.in_combat}")
        
        GameStateLogger._log_available_actions(game_state)
        
        if game_state.in_combat:
            GameStateLogger._log_combat_state(game_state)
        
        if game_state.screen_type == ScreenType.SHOP_SCREEN:
            GameStateLogger._log_shop_info(game_state)
            
        logger.info("=== END GAME STATE ===\n")

    @staticmethod
    def _log_available_actions(game_state: Game):
        """Log available actions"""
        logger.info("\nAvailable Actions:")
        logger.info(f"- Proceed available: {game_state.proceed_available}")
        logger.info(f"- Cancel available: {game_state.cancel_available}")
        logger.info(f"- Choice available: {game_state.choice_available}")
        
        if game_state.choice_available and hasattr(game_state, 'choice_list'):
            logger.info("\nAvailable Choices:")
            for i, choice in enumerate(game_state.choice_list):
                logger.info(f"  {i}: {choice}")

    @staticmethod
    def _log_combat_state(game_state: Game):
        """Log combat-specific information"""
        logger.info("\nCombat State:")
        logger.info(f"Player Block: {game_state.player.block}")
        logger.info(f"Player Energy: {game_state.player.energy}")
        logger.info("\nMonsters:")
        for i, monster in enumerate(game_state.monsters):
            logger.info(f"  Monster {i+1}: {monster.name}")
            logger.info(f"    HP: {monster.current_hp}/{monster.max_hp}")
            logger.info(f"    Block: {monster.block}")
            logger.info(f"    Intent: {monster.intent}")
            logger.info(f"    Is Gone: {monster.is_gone}")

    @staticmethod
    def _log_shop_info(game_state: Game):
        """Log shop-specific information"""
        logger.info("\nShop Information:")
        if hasattr(game_state.screen, 'cards'):
            logger.info("Cards for sale:")
            for card in game_state.screen.cards:
                logger.info(f"  - {card.name} (Cost: {card.price})")
        
        if hasattr(game_state.screen, 'relics'):
            logger.info("Relics for sale:")
            for relic in game_state.screen.relics:
                logger.info(f"  - {relic.name} (Cost: {relic.price})")
        
        if hasattr(game_state.screen, 'purge_available'):
            logger.info(f"Purge available: {game_state.screen.purge_available}")
            if game_state.screen.purge_available:
                logger.info(f"Purge cost: {game_state.screen.purge_cost}")

    @staticmethod
    def log_shop_visit(visited: bool):
        """Log shop visit state"""
        if visited:
            logger.info("First time visiting shop - choosing shopkeeper")
        else:
            logger.info("Already visited shop - proceeding")

    @staticmethod
    def log_error(error_msg: str):
        """Log error messages"""
        logger.error(f"Error encountered: {error_msg}")

    @staticmethod
    def log_game_progress(game_number: int, victory: bool, total_damage: int, win_rate: float, epsilon: float):
        """Log game progress information"""
        logger.info(f"\n=== Game {game_number} Summary ===")
        logger.info(f"Victory: {victory}")
        logger.info(f"Total damage taken: {total_damage}")
        logger.info(f"Win rate: {win_rate:.2f}%")
        logger.info(f"Epsilon: {epsilon:.3f}")
        logger.info("=== End Summary ===\n")