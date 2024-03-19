from soulsgym.games.game import Game  # noqa: TC001, required to avoid circular import errors
from soulsgym.games.darksouls3 import DarkSoulsIII
from soulsgym.games.eldenring import EldenRing
from soulsgym.games.tekken8 import Tekken8


def game_factory(game_id: str) -> Game:
    """Factory function for creating game interfaces.

    Args:
        game_id: The name of the game.

    Returns:
        The game interface.
    """
    match game_id:
        case "DarkSoulsIII":
            return DarkSoulsIII()
        case "EldenRing":
            return EldenRing()
        case "Tekken8":
            return Tekken8()
    raise ValueError(f"Unknown game: {game_id}")
