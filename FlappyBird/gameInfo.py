"""
    File name: gameInfo.py
    Author: Nikola Zubic
"""

from ple.games.flappybird import FlappyBird

game = FlappyBird()

SCREEN_WIDTH = game.getScreenDims()[0]
SCREEN_HEIGHT = game.getScreenDims()[1]
"""
When we perform "JUMP" action, we don't have to jump at exact pixel in order to surpass the pipes.
So we can approximate 15 pixels that are around resulting in smaller values.
"""
PIXEL_DISCOUNT = 15
MAX_DROP_SPEED = 19
