import gymnasium
import soulsgym
import msvcrt

if __name__ == "__main__":
    env = gymnasium.make("Tekken8", game_speed = 2.5)
    obs, info = env.reset()
    terminated = False
    while not terminated:
        if msvcrt.kbhit():  # Check if a key has been pressed
            key = msvcrt.getch()
            if key == b'p':  # Check if the key is 'p'
                print("Terminating the gym environment.")
                terminated = True
                break
        next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()