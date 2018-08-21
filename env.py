from __future__ import division, print_function
import numpy as np
import math
import os
import cv2
from itertools import cycle
from gym.spaces import Box, Discrete

SCREENWIDTH  = 288
SCREENHEIGHT = 512

PIPEGAPSIZE = 100 # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = 34
PLAYER_HEIGHT = 24

PIPE_WIDTH = 52
PIPE_HEIGHT = 320

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

class FlappyBirdEnv(object):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()

        # RL environment attributes
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=255, shape=(288, 512, 3), dtype=np.uint8)
        self.reward_range = (-math.inf, math.inf)

        # Game assets
        self.player = (
            cv2.imread(os.path.join('assets', 'sprites', 'redbird-upflap.png'), -1),
            cv2.imread(os.path.join('assets', 'sprites', 'redbird-midflap.png'), -1),
            cv2.imread(os.path.join('assets', 'sprites', 'redbird-downflap.png'), -1)
        )
        self.background = cv2.imread(os.path.join('assets', 'sprites', 'background-black.png'))
        self.pipe = cv2.imread(os.path.join('assets', 'sprites', 'pipe-green.png'), -1)
        self.base = cv2.imread(os.path.join('assets', 'sprites', 'base.png'))

        # Dynamics config
        self.pipeVelX = -4       # player's velocity along Y
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1      # players downward accleration
        self.playerFlapAcc = -9  # players speed on flapping
        self.baseShift = self.base.shape[1] - SCREENWIDTH

        self.reset()

    def reset(self):
        self.timestep = 0
        self.playerIndex = 0
        self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2.0)
        self.basex = 0

        new_pipes = [
            get_random_pipe(pipeX=SCREENWIDTH),
            get_random_pipe(pipeX=SCREENWIDTH + (SCREENWIDTH / 2))
        ]
        self.upper_pipes = list(map(lambda x: x['upper_pipe'], new_pipes))
        self.lower_pipes = list(map(lambda x: x['lower_pipe'], new_pipes))

        self.playerVelY = 0         # player's velocity along Y, default same as playerFlapped

        return self.render(), self.get_info()

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
                self.score += 1
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
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upper_pipes, self.lower_pipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upper_pipes[0]['x'] < 5:
            new_pipe = get_random_pipe()
            self.upper_pipes.append(new_pipe['upper_pipe'])
            self.lower_pipes.append(new_pipe['lower_pipe'])

        # remove first pipe if its out of the screen
        if self.upper_pipes[0]['x'] < -PIPE_WIDTH:
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        # check if crash here
        isCrash= checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        if check_crash(
            playerx=self.playerx,
            playery=self.playery,
            upper_pipes=self.upper_pipes,
            lower_pipes=self.lower_pipes,
            player_hitbox=self.player[self.playerIndex][:, :, 3] > 0,
            pipe_hitbox=self.pipe[:, :, ] > 0
        ):
            terminal = True
            self.__init__()
            reward = -1

    def render(self):
        return None

    def get_segmentation(self):
        return None

    def self.get_info(self):
        info = dict()
        info['playerx'] = self.playerx
        info['playery'] = self.playery
        info['upper_pipes'] = self.upper_pipes
        info['lower_pipes'] = self.lower_pipes
        info['playerVelX'] = -self.pipeVelX
        info['playerVelY'] = self.playerVelY
        info['segmentation'] = self.get_segmentation()
        return info

def check_crash(playerx, playery, upper_pipes, lower_pipes, player_hitbox, pipe_hitbox):
    return True

def get_random_pipe(pipeX=None):
    gapY = np.random.choice([20, 30, 40, 50, 60, 70, 80, 90])

    gapY += int(BASEY * 0.2)
    if pipeX is None:
        pipeX = SCREENWIDTH + 10

    return {
        'upper_pipe': {'x': pipeX, 'y': gapY - PIPE_HEIGHT},
        'lower_pipe': {'x': pipeX, 'y': gapY + PIPEGAPSIZE}
    }

if __name__ == '__main__':
    env = FlappyBirdEnv()
