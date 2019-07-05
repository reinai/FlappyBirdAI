"""
    File name: FlappyBirdDummyAgent.py
    Author: Nikola Zubic
"""

import numpy as np
NUM_STEPS = 15000


class NaiveAgent(object):
    def __init__(self, actions):
        self.actions = actions

    def pick_action(self):
        return self.actions[np.random.randint(0, len(self.actions))]


def run_a_game(game):
    from ple import PLE
    p = PLE(game, display_screen=True)
    agent = NaiveAgent(p.getActionSet())
    p.init()
    for i in range(NUM_STEPS):
        p.act(agent.pick_action())


def test_flappybird():
    from ple.games.flappybird import FlappyBird
    game = FlappyBird()
    run_a_game(game)


test_flappybird()
