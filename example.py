from Flappy_Bird_with_Segmentation.environment import FlappyBirdSegEnv
import numpy as np
import cv2
import imageio


def expert_action(info):
    pipe_idx = 0
    while info['lower_pipes'][pipe_idx]['x'] + 52 - info['playerVelX'] <= info['playerx']:
        pipe_idx += 1
    return info['playery'] + 24 + info['playerVelY'] + 1 >= info['lower_pipes'][pipe_idx]['y']


def draw_seg(array):
    classes = {
        0: [26, 26, 26],
        1: [149, 216, 222],
        2: [0, 255, 0],
        3: [0, 56, 252]
    }

    result = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result


if __name__ == '__main__':
    env = FlappyBirdSegEnv()
    obs, info = env.reset()
    frames = []
    for i in range(200):
        obs, reward, terminal, info = env.step(expert_action(info))
        frames.append(cv2.cvtColor(np.concatenate([obs, draw_seg(info['segmentation'])], axis=1), cv2.COLOR_BGR2RGB))
        if terminal:
            break
    imageio.mimsave("images/demo.gif", frames, fps=24)
