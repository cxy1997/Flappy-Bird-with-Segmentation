from __future__ import division, print_function
import numpy as np
import math
import os
import cv2
from itertools import cycle
from gym.spaces import Box, Discrete

SCREENWIDTH = 288
SCREENHEIGHT = 512

PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
BASEY = int(SCREENHEIGHT * 0.79)

PLAYER_WIDTH = 34
PLAYER_HEIGHT = 24

PIPE_WIDTH = 52
PIPE_HEIGHT = 320

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


class FlappyBirdSegEnv(object):
    def __init__(self):
        super(FlappyBirdSegEnv, self).__init__()

        # RL environment attributes
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=255, shape=(288, 512, 3), dtype=np.uint8)
        self.reward_range = (-math.inf, math.inf)

        # Game assets
        self.player = (
            cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'sprites', 'redbird-upflap.png'), -1),
            cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'sprites', 'redbird-midflap.png'), -1),
            cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'sprites', 'redbird-downflap.png'), -1)
        )
        self.background = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'sprites', 'background-black.png'))
        self.lowerpipe = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'sprites', 'pipe-green.png'), -1)
        self.upperpipe = np.rot90(self.lowerpipe, k=2)
        self.base = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'sprites', 'base.png'), -1)

        # Dynamics config
        self.pipeVelX = -4       # player's velocity along Y
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1      # players downward accleration
        self.playerFlapAcc = -9  # players speed on flapping
        self.baseShift = self.base.shape[1] - SCREENWIDTH

    def reset(self):
        self.timestep = 0
        self.playerIndex = 0
        self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2.0)
        self.basex = 0

        new_pipes = [
            get_random_pipe(pipeX=SCREENWIDTH - PIPE_WIDTH - 3),
            get_random_pipe(pipeX=SCREENWIDTH - PIPE_WIDTH - 3 + int(SCREENWIDTH / 2))
        ]
        self.upper_pipes = list(map(lambda x: x['upper_pipe'], new_pipes))
        self.lower_pipes = list(map(lambda x: x['lower_pipe'], new_pipes))

        self.playerVelY = 0  # player's velocity along Y, default same as playerFlapped

        obs, self.segmentation, self.collision = self.render()
        self.collision = self.collision or self.playery >= BASEY - PLAYER_HEIGHT
        return obs, self.get_info()

    def step(self, action):
        self.timestep += 1
        action = int(action)
        reward = 0.1
        terminal = False

        if action == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc

        # check for score
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upper_pipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                reward = 1
                break

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and action == 0:
            self.playerVelY += self.playerAccY
        self.playery = np.clip(self.playery + self.playerVelY, 0, BASEY - PLAYER_HEIGHT)

        # move pipes to left
        for uPipe, lPipe in zip(self.upper_pipes, self.lower_pipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upper_pipes[0]['x'] < -self.pipeVelX:
            new_pipe = get_random_pipe()
            self.upper_pipes.append(new_pipe['upper_pipe'])
            self.lower_pipes.append(new_pipe['lower_pipe'])

        # remove first pipe if its out of the screen
        if self.upper_pipes[0]['x'] <= -PIPE_WIDTH:
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        obs, self.segmentation, self.collision = self.render()
        self.collision = self.collision or self.playery >= BASEY - PLAYER_HEIGHT
        if self.collision:
            terminal = True
            reward = -1
        return obs, reward, terminal, self.get_info()

    def render(self):
        obs = np.array(self.background, dtype=np.uint8)
        seg = np.zeros(self.background.shape[:-1], dtype=np.uint8)
        obs, seg, _ = draw_upon(obs, seg, self.base, self.basex, BASEY, 1)
        for upperpipe, lowerpipe in zip(self.upper_pipes, self.lower_pipes):
            if -PIPE_WIDTH < upperpipe['x'] < SCREENWIDTH:
                obs, seg, _ = draw_upon(obs, seg, self.upperpipe, upperpipe['x'], upperpipe['y'], 2)
                obs, seg, _ = draw_upon(obs, seg, self.lowerpipe, lowerpipe['x'], lowerpipe['y'], 2)
        obs, seg, collision = draw_upon(obs, seg, self.player[self.playerIndex], self.playerx, self.playery, 3)
        return obs, seg, collision

    def get_info(self):
        return {
            'playerx': self.playerx,
            'playery': self.playery,
            'upper_pipes': self.upper_pipes,
            'lower_pipes': self.lower_pipes,
            'playerVelX': -self.pipeVelX,
            'playerVelY': self.playerVelY,
            'segmentation': self.segmentation,
            'collision': self.collision
        }


def draw_upon(obs, seg, obj, x, y, classID):
    x, y = y, x
    H, W, C = obs.shape
    assert C == 3
    h, w, c = obj.shape
    assert c == 3 or c == 4
    if c == 3:
        obj = np.concatenate([obj, 255 * np.ones((h, w), dtype=np.uint8)], axis=2)
    obj_x_start, obj_x_end = max(-x, 0), min(h, H-x)
    obj_y_start, obj_y_end = max(-y, 0), min(w, W-y)

    patch = obj[obj_x_start: obj_x_end, obj_y_start: obj_y_end, :3]
    seg_mask = obj[obj_x_start: obj_x_end, obj_y_start: obj_y_end, 3] > 0
    collision = np.any(seg_mask * seg[x+obj_x_start: x+obj_x_end, y+obj_y_start: y+obj_y_end] > 0) if classID == 3 else None
    rgb_mask = np.expand_dims(seg_mask, axis=2)
    obs[x+obj_x_start: x+obj_x_end, y+obj_y_start: y+obj_y_end, :3] = patch * rgb_mask + obs[x+obj_x_start: x+obj_x_end, y+obj_y_start: y+obj_y_end, :3] * (1-rgb_mask)
    seg[x+obj_x_start: x+obj_x_end, y+obj_y_start: y+obj_y_end] = classID * seg_mask + seg[x+obj_x_start: x+obj_x_end, y+obj_y_start: y+obj_y_end] * (1-seg_mask)
    return obs, seg, collision


def get_random_pipe(pipeX=None):
    gapY = np.random.choice([20, 30, 40, 50, 60, 70, 80, 90])

    gapY += int(BASEY * 0.2)
    if pipeX is None:
        pipeX = SCREENWIDTH + 10

    return {
        'upper_pipe': {'x': pipeX, 'y': gapY - PIPE_HEIGHT},
        'lower_pipe': {'x': pipeX, 'y': gapY + PIPEGAPSIZE}
    }
