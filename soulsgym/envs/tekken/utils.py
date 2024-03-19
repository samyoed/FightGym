
from soulsgym.core.game_input import GameInput


def reset_practice():
    game_input = GameInput(game_id="MenuTekken8")
    game_input.multi_action(["three","start"], press_time=0.1)