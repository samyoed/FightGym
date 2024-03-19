import logging
import struct
from soulsgym.core.game_window import GameWindow
from typing import Any
import time

from pymem.exception import MemoryReadError
import numpy as np

from soulsgym.games import Game
from soulsgym.core.utils import wrap_to_pi
import cv2

logger = logging.getLogger(__name__)


class Tekken8(Game):

    game_id = "Tekken8"
    process_name = "TEKKEN 8.exe"
    leftHPCapture = {'top': 5, 'left': 16, 'width': 55, 'height': 4}
    rightHPCapture = {'top': 5, 'left': 88, 'width': 55, 'height': 4}
    game_window = None


    def __init__(self):
        super().__init__()
        self._game_flags = {}  # Cache game flags to restore them after a game reload
        self._game_speed = 1.0
        self.game_speed = 1.0
        self._game_window.focus_application()
        self.game_window = self._game_window.get_img()

    @property
    def img(self) -> np.ndarray:
        """Get the current game image as numpy array.

        Warning:
            If the game was paused (i.e. ``game_speed = 0``) before the current ```Game`` instance
            has been created, this method won't return. The game needs to be unpaused at least once
            before invoking this method.

        Images have a shape of [90, 160, 3] with RGB channels.
        """
        # self.show_window()
        self.game_window = self._game_window.get_img()
        return self.game_window

    @img.setter
    def img(self, _: Any):
        raise RuntimeError("Game image can't be set!")

    @property
    def img_resolution(self) -> tuple[int, int]:
        """The game image resolution.

        Note:
            This is NOT the game resolution. The game image resolution is the resolution of the
            image returned by `:meth:.EldenRing.img`.

        Returns:
            The game resolution.
        """
        return self._game_window.resolution

    @img_resolution.setter
    def img_resolution(self, resolution: tuple[int, int]):
        self._game_window.img_resolution = resolution

    @property
    def hp_frame_left(self) -> int:
        """The left player's current hp frame

        Returns:
            The left player's current hp frame
        """
        if self.game_window is None:
            self.game_window = self._game_window.get_img()

        hp_frame = self.game_window[self.leftHPCapture['top']:self.leftHPCapture['top']+self.leftHPCapture['height'],
                         self.leftHPCapture['left']:self.leftHPCapture['left']+self.leftHPCapture['width']]
        hp_frame = np.dot(hp_frame[...,:3], [0.299,0.587,0.114])
        
        # Display the HP frame
        
        return hp_frame
    
    @property
    def hp_frame_right(self) -> int:
        """The right player's current hp frame

        Returns:
            The right player's current hp frame
        """
        if self.game_window is None:
            self.game_window = self._game_window.get_img()

        hp_frame = self.game_window[self.rightHPCapture['top']:self.rightHPCapture['top']+self.rightHPCapture['height'],
                           self.rightHPCapture['left']:self.rightHPCapture['left']+self.rightHPCapture['width']]
        hp_frame = np.dot(hp_frame[...,:3], [0.299,0.587,0.114])
        return hp_frame
    
    def reload(self):
        """use controller to reset round"""
        raise NotImplementedError("Respawn animations need to be checked!")

    @property
    def side(self) -> int:
        """The current player's side.

        Returns:
            The current player's side.
        """
        return 0
    
    @property
    def char_left(self) -> int:
        """The left player's character.

        Returns:
            The left player's character.
        """
        return 0
    
    @property
    def char_right(self) -> int:
        """The right player's character.

        Returns:
            The right player's character.
        """
        return 1
    
    @property
    def char(self) -> int:
        """The current player's character.

        Returns:
            The current player's character.
        """
        if self.side == 0:
            return self.char_left
        else:
            return self.char_right
    
    @property
    def stage(self) -> int:
        """The current stage.

        Returns:
            The current stage.
        """
        return 0


    @property
    def time(self) -> int:
        """Ingame time.

        either try to read the time from screen or just don't do it

        Returns:
            The current game time.
        """
        # get the difference between the current time and the time when the game was started

        return 0

    @staticmethod
    def timed(tend: int, tstart: int) -> float:
        """Safe game time difference function.

        If time has overflowed, uses 0 as best guess for tstart. Divides by 1000 to get the time
        difference in seconds.

        Args:
            tend: End time.
            tstart: Start time.

        Returns:
            The time difference.
        """
        return (tend - tstart) / 1000 if tend >= tstart else tend / 1000

    def sleep(self, t: float):
        """Custom sleep function.

        Guarantees the specified time has passed in ingame time.

        Args:
            t: Time interval in seconds.
        """
        assert t > 0
        assert self.game_speed > 0, "Game can't be paused during sleeps"
        # We save the start time and use nonbusy python sleeps while t has not been reached
        tstart, td = self.time, t
        while True:
            time.sleep(td / self.game_speed)
            tcurr = self.time
            if self.timed(tcurr, tstart) > t:
                break
            # 1e-3 / game_speed is the min waiting interval
            td = max(t - self.timed(tcurr, tstart), 1e-3) / self.game_speed

    @property
    def game_speed(self) -> float:
        """The game loop speed.

        Note:
            Setting this value to 0 will effectively pause the game. Default speed is 1.

        Warning:
            The process slows down with game speeds lower than 1. Values close to 0 may cause
            windows to assume the process has frozen.

        Warning:
            Values significantly higher than 1 (e.g. 5+) may not be achievable for the game loop.
            This is probably dependant on the available hardware.

        Returns:
            The game loop speed.

        Raises:
            ValueError: The game speed was set to negative values.
        """
        return self._game_speed

    @game_speed.setter
    def game_speed(self, value: float):
        if value < 0:
            raise ValueError("Attempting to set a negative game speed")
        self._speed_hack_connector.set_game_speed(value)
        self._game_speed = value
