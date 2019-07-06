"""
    File name: monteCarloQLearningAgent.py
    Author: Nikola Zubic
"""

import random
import math
from FlappyBird.gameInfo import *
from ple import PLE
from ple.games.flappybird import FlappyBird

from monteCarloSearch import Node
from monteCarloSearch import MonteCarloSearch

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
NUMBER_OF_SIMULATIONS = 60

def flip_coin(p):
    r = random.random()
    return r > p


class IllegalActionException(BaseException):
    def __init__(self, message):
        print(message)


game = FlappyBird()
env = PLE(game=game, fps=30, display_screen=False, force_fps=True)


def children_search(node, monte_carlo):
    depth = 200
    state = node.get_state()
    while depth > 0:
        state = node.get_best_successors()
        node.add_child(state)
        depth -= 1

def compute_node_values(monte_carlo, node):
    state = node.get_state()
    game_over = env.game_over()
    if game_over:
        return -5
    else:
        return 1


class MonteCarloQLearningAgent(object):
    """
      Monte Carlo Q-Learning Agent

      Functions:
        - __init__ constructor
        - is_state_action_explored
        - max_q
        - compute_action_from_q_values
        - update
        - get_state (static)
    """

    def __init__(self):
        self.epsilon = 0.1
        self.Q_matrix = {}

    def is_state_action_explored(self, state):
        """
        For pair (state, action) we check if it is available in Q matrix.
        In other words, have we been in state "state" and decided to perform action A = {0, 1}
        """
        state_action_up = (state, 0)  # for "UP" action
        state_action_steady = (state, 1)  # no key pressed

        if state_action_up in self.Q_matrix and state_action_steady in self.Q_matrix:
            return True
        else:
            if state_action_up not in self.Q_matrix:
                return False, 0
            elif state_action_steady not in self.Q_matrix:
                return False, 1

    def max_q(self, state):
        """
        Idea is to pick the action with biggest (best) Q-Value of next state in order
        to stay in optimal policy Qpi* .
        So, if policy of SARSA is greedy, it becomes Q-learning.
        """

        if (state, 0) not in self.Q_matrix:
            self.Q_matrix[(state, 0)] = 0
        if (state, 1) not in self.Q_matrix:
            self.Q_matrix[(state, 1)] = 0

        if self.Q_matrix[(state, 0)] > self.Q_matrix[(state, 1)]:
            return 0
        elif self.Q_matrix[(state, 0)] < self.Q_matrix[(state, 1)]:
            return 1
        else:
            return random.randint(0, 1)

    def compute_action_from_q_values(self, state):
        """
          https://arxiv.org/pdf/1802.05944.pdf
        """
        try:

            if (state, 0) not in self.Q_matrix:
                self.Q_matrix[(state, 0)] = 0
            if (state, 1) not in self.Q_matrix:
                self.Q_matrix[(state, 1)] = 0

            if flip_coin(self.epsilon):
                monte_carlo = MonteCarloSearch(Node(env.game.getGameState()))

                monte_carlo.children_search = children_search
                monte_carlo.compute_node_values = compute_node_values(monte_carlo, Node(env.game.getGameState()))

                monte_carlo.simulation(NUMBER_OF_SIMULATIONS)

                monte_carlo.selection()

                # We will select action with highest Q-Value
                if self.Q_matrix[(state, 0)] > self.Q_matrix[(state, 1)]:
                    return 0
                elif self.Q_matrix[(state, 1)] > self.Q_matrix[(state, 0)]:
                    return 1
                else:
                    return random.randint(0, 1)
            else:
                # We will do exploration, by selecting the random action
                return random.randint(0, 1)

        except IllegalActionException:
            return None

    def update(self, state, action, next_state, reward):
        """
          We call this to observe a
          state = action => nextState and reward transition.
          Here is done the Q-Value update.

          For update, we use formula given here within the algorithm:
            https://nthu-datalab.github.io/ml/labs/16-1_Q-Learning/16-1_Q_Learning.html

        """
        sample = (reward + DISCOUNT_FACTOR * self.Q_matrix[(next_state, self.max_q(next_state))] -
                  self.Q_matrix[(state, action)])

        self.Q_matrix[(state, action)] = self.Q_matrix[(state, action)] + LEARNING_RATE * sample

    @staticmethod
    def get_state(state):

        next_pipe_distance_to_player = math.floor(state['next_pipe_dist_to_player'] * bucket_range_per_feature /
                                                  SCREEN_WIDTH)
        player_vel = math.floor((state['player_vel'] - MIN_VELOCITY) * FIRST_NON_ZERO_VELOCITY / MAX_VELOCITY)

        player_y = state['player_y']
        pipe_gap_average = (state['next_pipe_top_y'] + state['next_pipe_bottom_y']) / 2.0
        next_pipe_gap_y_distance_to_player = math.floor((player_y - pipe_gap_average) * bucket_range_per_feature /
                                                        SCREEN_HEIGHT)

        state = (player_vel, next_pipe_distance_to_player, next_pipe_gap_y_distance_to_player)

        return state


agent = MonteCarloQLearningAgent()
