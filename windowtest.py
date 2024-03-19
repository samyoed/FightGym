from typing import Callable
import cv2
import numpy as np
import time
import platform

# Ensure this script runs on Windows
if platform.system() != "Windows":
    raise SystemExit("This script only works on Windows.")

from soulsgym.core.game_window.game_window import GameWindow

def capture_real_time_footage():
    try:
        # Initialize the GameWindow for "Dark Souls III"
        game_window = GameWindow(game_id="Tekken8")
        
        # Focus the game application
        game_window.focus_application()
        time.sleep(1)  # Wait a bit for the window to focus

        # Open a window to display the footage
        cv2.namedWindow("Real-Time Footage", cv2.WINDOW_NORMAL)

        frame = game_window.get_img()
        screen = {'top': 0, 'left': 0, 'width': 160, 'height': 90}
        leftHPCapture = {'top': 5, 'left': 16, 'width': 55, 'height': 4}
        rightHPCapture = {'top': 5, 'left': 88, 'width': 55, 'height': 4}

        prevLeft = frame[leftHPCapture['top']:leftHPCapture['top']+leftHPCapture['height'],
                          leftHPCapture['left']:leftHPCapture['left']+leftHPCapture['width']]
        prevRight = frame[rightHPCapture['top']:rightHPCapture['top']+rightHPCapture['height'],
                           rightHPCapture['left']:rightHPCapture['left']+rightHPCapture['width']]


        prevLeft = np.dot(prevLeft[...,:3], [0.299,0.587,0.114])
        prevRight = np.dot(prevRight[...,:3], [0.299,0.587,0.114])

        while True:
            # Capture a framepp
            frame = game_window.get_img()

            currLeft = frame[leftHPCapture['top']:leftHPCapture['top']+leftHPCapture['height'],
                              leftHPCapture['left']:leftHPCapture['left']+leftHPCapture['width']]
            currRight = frame[rightHPCapture['top']:rightHPCapture['top']+rightHPCapture['height'],
                               rightHPCapture['left']:rightHPCapture['left']+rightHPCapture['width']]
            currLeft = np.dot(currLeft[...,:3], [0.299,0.587,0.114])
            currRight = np.dot(currRight[...,:3], [0.299,0.587,0.114])

            diffLeft = prevLeft - currLeft
            diffRight = prevRight - currRight

            diffLeft = diffLeft.clip(min=0)
            diffRight = diffRight.clip(min=0)
            # Check for hit on left
            if((diffLeft > 125).sum() > 5):
                print('Left was hit')
            # Check for hit on right
            if((diffRight > 125).sum() > 5):
                print('Right was hit')


            print(f"right {(diffRight > 125).sum()}")
            print(f"left {(diffLeft > 125).sum()}")


            cv2.rectangle(frame, (leftHPCapture['left'], leftHPCapture['top']-screen['top']), 
                          (leftHPCapture['left'] + leftHPCapture['width'], 
                           leftHPCapture['top']-screen['top'] + leftHPCapture['height']), (0, 255, 0), 1)
            cv2.rectangle(frame, (rightHPCapture['left'], rightHPCapture['top']-screen['top']), 
                          (rightHPCapture['left'] + rightHPCapture['width'], 
                           rightHPCapture['top']-screen['top'] + rightHPCapture['height']), (0, 255, 0), 1)
                
            # Display the frame
            cv2.imshow("Real-Time Footage", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # When everything done, release the capture
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    capture_real_time_footage()
