from __future__ import annotations

import logging
import random
import time
from typing import Any, TYPE_CHECKING
from dataclasses import dataclass, field, asdict

import numpy as np
from gymnasium import spaces

from soulsgym.envs.game_state import GameState
from soulsgym.envs.soulsenv import SoulsEnv, SoulsEnvDemo
from soulsgym.envs.utils import max_retries
from soulsgym.exception import GameStateError, ResetError
from soulsgym.envs.tekken.utils import reset_practice

if TYPE_CHECKING:
    from soulsgym.games import Tekken8

logger = logging.getLogger(__name__)

@dataclass
class TekkenState(GameState):
    """Collect all game information for state tracking in a single data class.

    This class extends the base ``GameState`` with additional data members that are specific to the
    Iudex Gundyr fight.
    """
    side: int = 0
    char_left: int = 0
    char_right: int = 0
    char: int = 0
    stage: int = 0
    hp_frame_left: np.ndarray = field(default_factory=lambda: np.zeros((4, 55), dtype=np.uint8))
    hp_frame_right: np.ndarray = field(default_factory=lambda: np.zeros((4, 55), dtype=np.uint8))
    


    """override"""
    def copy(self) -> 'TekkenState':
        """Create a deep copy of the ``TekkenState``.

        Returns:
            A deep copy of itself.
        """
        # Manually copy each field to ensure a deep copy, especially for the numpy arrays
        new_copy = TekkenState(
            side=self.side,
            char_left=self.char_left,
            char_right=self.char_right,
            char=self.char,
            stage=self.stage,
            # Use the numpy array copy method for a deep copy of these fields
            hp_frame_left=self.hp_frame_left.copy(),
            hp_frame_right=self.hp_frame_right.copy()
        )
        return new_copy
    
    @staticmethod
    def from_dict(data_dict: dict) -> GameState:
        """Create a ``GameState`` object from a dictionary.

        Args:
            data_dict: Dictionary containing the GameState information.

        Returns:
            A GameState object with matching values.
        """
        for key, value in data_dict.items():
            if isinstance(value, list):
                data_dict[key] = np.array(value)
        return TekkenState(**data_dict)
    
class Tekkenv(SoulsEnv):
    """Gymnasium environment class for the Iudex Gundyr bossfight.

    The environment uses the ground truth information from the game as observation space.
    """


    def __init__(self, game_speed: float = 1.0, resolution: tuple[int, int] = (90, 160)):
        """Initialize the observation and action spaces.

        Args:
            game_speed: Determines how fast the game runs during :meth:`.SoulsEnv.step`.
        """
        super().__init__(game_speed=game_speed)
        self.game: Tekken8  # Type hint only
        self._game_state: TekkenState  # Type hint only
        assert len(resolution) == 2
        self.observation_space = spaces.Box(low=0,high=255,shape=resolution + (3,),dtype=np.uint8)
        self.game.img_resolution = resolution
        self.action_space = spaces.Discrete(len(self.game.data.actions))
        self.step_count: int = 0
        self._arena_setup()

    @property
    def game_id(self) -> str:
        """Return the ID of the souls game that is required to run for this environment.

        Returns:
            The game ID.
        """
        return "Tekken8"

    @property
    def obs(self) -> dict:
        """Return the current observation."""
        return self.game.img
    @property
    def info(self) -> dict:
        """Info property of the environment.

        Returns:
            The current info dict of the environment.
        """
        return {
        }
    
    def _apply_action(self, action: int):
        """Apply an action to the environment.
        If the player is currently in an animation where he is disabled we omit all actions. The
        game queues actions and performs them as soon as the player is able to move. If an agent
        takes action 1 during the first step while being disabled, action 1 might be executed during
        the next step even though the agent has chosen action 2. In particular, roll and hit
        commands overwrite any normal movement. As long as we can't clear this queue we have to
        ensure actions are only performed when possible. Since we do not have access to the disabled
        game flags for the player we measure the duration of the current animation and compare that
        to the experimentally determined timeframe for this animation during which the player is
        disabled.
        Args:
            action: The action that is applied during this step.
        """
        self._game_input.add_actions(self.game.data.actions[action])
        # We always call the update because it includes actions added by _lock_on for camera control
        # If no action was queued, the update is equivalent to a reset.
        self._game_input.update_input()

    def _step(self, action: int):
        """Perform the actual step ingame.

        Unpauses the game, takes 0.01s substeps ingame, checks if the step size is already reached,
        times animations, handles critical events, updates the internal state and resets the player
        and boss HP. Once the ``step_size`` length has been reached the game gets paused again and
        step postprocessing begins.

        Args:
            action: The action that is applied during this step.
        """
        self.game.game_speed = self._game_speed
        # Needs to be called AFTER resume game to apply roll/hits. Since roll and hit actions have
        # a blocking sleep call, we also begin the timing of animations before applying the action
        # so that this sleep is accounted for in the total step time.
        self._apply_action(action)
        # Offset of 0.01s to account for processing time of the loop
            # Theoretically limits the loop to 1000 iterations / step. Effectively reduces the loop
            # to a few iterations as context switching allows the CPU to schedule other processes.
            # Disabled for now to increase loop timing precision
            # time.sleep(self.step_size / 1000.)
        time.sleep(self.step_size/self._game_speed)
        # self.step_count +=1
        # print(self.step_count)
        self._game_state = self.game_state()
    def game_state(self) -> TekkenState:
        """Read the current game state.

        Returns:
            The current game state.
        """
        game_state = TekkenState()
        game_state.hp_frame_left = self.game.hp_frame_left
        game_state.hp_frame_right = self.game.hp_frame_right
        game_state.side = self.game.side
        game_state.char_left = self.game.char_left
        game_state.char_right = self.game.char_right
        game_state.char = self.game.char
        return game_state.copy()

    @max_retries(retries=3)
    def reset(self, seed: int | None = None, options: Any | None = None) -> tuple[dict, dict]:
        """Reset the environment to the beginning of an episode.

        Args:
            seed: Random seed. Required by gymnasium, but does not apply to SoulsGyms.
            options: Options argument required by gymnasium. Not used in SoulsGym.

        Returns:
            A tuple of the first game state and the info dict after the reset.
        """
        self._game_state = self.game_state()
        self._arena_setup()
        return self.obs, self.info


    @max_retries(retries=5)
    def _arena_setup(self):
        self.game.game_speed = 2.5
        reset_practice()
        self._game_input.reset()
        
        """Set up the arena.

        Args:
            init_retries: Maximum number of retries in case of initialization failure.
        """

    @staticmethod
    def compute_reward(game_state: TekkenState, next_game_state: TekkenState) -> float:
        """Compute the reward from the current game state and the next game state.

        Args:
            game_state: The game state before the step.
            next_game_state: The game state after the step.

        Returns:
            The reward for the provided game states.
        """
        reward = 0

        diff_left = next_game_state.hp_frame_left - game_state.hp_frame_left
        diff_right = next_game_state.hp_frame_right - game_state.hp_frame_right
        diff_left = diff_left.clip(min=0)
        diff_right = diff_right.clip(min=0)

    #    add reward for hitting the other player. depending on the side, the reward is different
        if game_state.side == 0:
            if((diff_left > 125).sum() > 5):
                reward -=.75
            # Check for hit on right
            if((diff_right > 125).sum() > 5):
                reward +=1
        else:
            if((diff_left > 125).sum() > 5):
                reward +=1
            # Check for hit on right
            if((diff_right > 125).sum() > 5):
                reward -=.75
        return reward

class TekkenvDemo(Tekkenv):
    """Demo environment for the Iudex Gundyr fight.

    Covers both phases. Player and boss loose HP, and the episode does not reset.
    """

    def __init__(self, game_speed: float = 1., random_player_pose: bool = False):
        """Initialize the demo environment.

        Args:
            game_speed: Determines how fast the game runs during :meth:`.SoulsEnv.step`.
            random_player_pose: Flag to randomize the player pose on reset.
        """
        super().__init__(game_speed)
        # IudexEnv can't be called with all arguments, so we have to set it manually after __init__

    def reset(self, seed: int | None = None, options: Any | None = None) -> tuple[dict, dict]:
        """Reset the environment to the beginning of an episode.

        Args:
            seed: Random seed. Required by gymnasium, but does not apply to SoulsGyms.
            options: Options argument required by gymnasium. Not used in SoulsGym.

        Returns:
            A tuple of the first game state and the info dict after the reset.
        """
        self._game_input.reset()
        self.game.reload()
        return super().reset()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        """Perform a step forward in the environment with a given action.

        Each step advances the ingame time by `step_size` seconds. The game is paused before and
        after the step.

        Args:
            action: The action that is applied during this step.

        Returns:
            A tuple of the next game state, the reward, the terminated flag, the truncated flag, and
            an additional info dictionary.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
