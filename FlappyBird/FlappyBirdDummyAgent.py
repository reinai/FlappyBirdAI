import numpy as np
NUM_STEPS = 15000

class NaiveAgent():
    def __init__(self, actions):
        self.actions = actions
    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]

def run_a_game(game):
    from ple import PLE
    p = PLE(game, display_screen=True)
    agent = NaiveAgent(p.getActionSet())
    p.init()
    reward = p.act(p.NOOP)
    for i in range(NUM_STEPS):
        obs = p.getScreenRGB()
        reward = p.act(agent.pickAction(reward, obs))

def test_flappybird():
    from ple.games.flappybird import FlappyBird
    game = FlappyBird()
    run_a_game(game)

test_flappybird()