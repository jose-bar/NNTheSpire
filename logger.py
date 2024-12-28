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
        if game_state is None:
            logger.error("Game state is None. Cannot log game state.")
            return

        logger.info("\n=== GAME STATE ===")
        logger.info(f"Screen Type: {getattr(game_state, 'screen_type', 'Unknown')}")
        
        GameStateLogger._log_available_actions(game_state)

        if getattr(game_state, 'in_combat', False):
            GameStateLogger._log_combat_state(game_state)
        
        if getattr(game_state, 'screen_type', None) == ScreenType.SHOP_SCREEN:
            GameStateLogger._log_shop_info(game_state)
            
        logger.info("=== END GAME STATE ===\n")

    @staticmethod
    def _log_available_actions(game_state: Game):
        """Log available actions"""
        logger.info("\nAvailable Actions:")
        logger.info(f"- Proceed available: {getattr(game_state, 'proceed_available', False)}")
        logger.info(f"- Cancel available: {getattr(game_state, 'cancel_available', False)}")
        logger.info(f"- Choice available: {getattr(game_state, 'choice_available', False)}")
        
        if getattr(game_state, 'choice_available', False) and hasattr(game_state, 'choice_list'):
            logger.info("\nAvailable Choices:")
            for i, choice in enumerate(game_state.choice_list):
                logger.info(f"  {i}: {choice}")

    @staticmethod
    def _log_combat_state(game_state: Game):
        """Log combat-specific information"""
        if not hasattr(game_state, 'player') or not hasattr(game_state, 'monsters'):
            logger.error("Combat state logging failed: missing player or monsters.")
            return

        logger.info("\nCombat State:")
        player = game_state.player
        logger.info(f"Player Block: {getattr(player, 'block', 0)}")
        logger.info(f"Player Energy: {getattr(player, 'energy', 0)}")

        logger.info("\nMonsters:")
        for i, monster in enumerate(getattr(game_state, 'monsters', [])):
            logger.info(f"  Monster {i+1}: {getattr(monster, 'name', 'Unknown')}")
            logger.info(f"    HP: {getattr(monster, 'current_hp', 0)}/{getattr(monster, 'max_hp', 0)}")
            logger.info(f"    Block: {getattr(monster, 'block', 0)}")
            logger.info(f"    Intent: {getattr(monster, 'intent', 'Unknown')}")
            logger.info(f"    Is Gone: {getattr(monster, 'is_gone', False)}")

    @staticmethod
    def _log_shop_info(game_state: Game):
        """Log shop-specific information"""
        screen = getattr(game_state, 'screen', None)
        if screen is None:
            logger.error("Shop screen information missing.")
            return

        logger.info("\nShop Information:")
        if hasattr(screen, 'cards'):
            logger.info("Cards for sale:")
            for card in screen.cards:
                logger.info(f"  - {getattr(card, 'name', 'Unknown')} (Cost: {getattr(card, 'price', 'Unknown')})")
        
        if hasattr(screen, 'relics'):
            logger.info("Relics for sale:")
            for relic in screen.relics:
                logger.info(f"  - {getattr(relic, 'name', 'Unknown')} (Cost: {getattr(relic, 'price', 'Unknown')})")
        
        if hasattr(screen, 'purge_available'):
            logger.info(f"Purge available: {screen.purge_available}")
            if screen.purge_available:
                logger.info(f"Purge cost: {getattr(screen, 'purge_cost', 'Unknown')}")

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
