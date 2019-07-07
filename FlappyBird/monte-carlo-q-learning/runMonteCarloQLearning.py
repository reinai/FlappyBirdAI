"""
    File name: runMonteCarloQLearning.py
    Author: Nikola Zubic
"""

from monteCarloQLearningAgent import MonteCarloQLearningAgent
from monteCarloQLearningAgent import IllegalActionException

from ple import PLE
from ple.games.flappybird import FlappyBird
import ast

monte_carlo_q_agent = MonteCarloQLearningAgent()
f = open("monte_150000.txt", "r")
monte_carlo_q_agent.Q_matrix = ast.literal_eval(f.read())

# f = open("agent_150000_results_medium.txt", "w")
# f.write("Monte Carlo Q learning agent after 150 000 episodes. Medium mode (pipe_gap = 120)\n")


def run(number_of_episodes):
    game = FlappyBird(pipe_gap=120)

    rewards = {
        "positive": 1.0,
        "negative": 0.0,
        "tick": 0.0,
        "loss": 0.0,
        "win": 0.0
    }

    env = PLE(game=game, fps=30, display_screen=True, reward_values=rewards, force_fps=False)

    # Reset environment at the beginning
    env.reset_game()

    score = 0
    max_score = 0
    episode_number = 1

    while number_of_episodes > 0:

        # Get current state
        state = MonteCarloQLearningAgent.get_state(env.game.getGameState())

        # Select action in state "state"
        action = monte_carlo_q_agent.max_q(state)

        """
        After choosing action, get reward.
        PLE environment method act() returns the reward that the agent has accumulated while performing the action.
        """
        reward = env.act(env.getActionSet()[action])
        score += reward

        max_score = max(score, max_score)

        game_over = env.game_over()

        if game_over:
            print("===========================")
            print("Episode: " + str(episode_number))
            print("Score: " + str(score))
            print("Max. score: " + str(max_score))
            print("===========================\n")
            # f.write("Score: " + str(score) + "|Max. score: " + str(max_score) + "\n")
            episode_number += 1
            number_of_episodes -= 1
            score = 0
            env.reset_game()


run(20)
