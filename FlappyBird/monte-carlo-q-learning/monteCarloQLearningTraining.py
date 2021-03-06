"""
    File name: monteCarloQLearningTraining.py
    Author: Nikola Zubic
"""
from monteCarloQLearningAgent import MonteCarloQLearningAgent
from monteCarloQLearningAgent import IllegalActionException

from ple import PLE
from ple.games.flappybird import FlappyBird

monte_carlo_q_agent = MonteCarloQLearningAgent()


def train_agent(number_of_episodes):
    game = FlappyBird()

    rewards = {
        "positive": 1.0,
        "negative": 0.0,
        "tick": 0.0,
        "loss": -5.0,
        "win": 0.0
    }

    env = PLE(game=game, fps=30, display_screen=False, reward_values=rewards)

    # Reset environment at the beginning
    env.reset_game()

    training_score = 0
    max_training_score = 0
    episode_number = 1

    state_action_reward = ()

    every_100th = 1
    results = []

    while number_of_episodes > 0:

        if episode_number == 50000:
            f = open("monte_50000.txt", "w")
            f.write(str(monte_carlo_q_agent.Q_matrix))
            f.close()
            f = open("results_50000.txt", "w")
            f.write(str(results))
            f.close()

        # Get current state
        state = MonteCarloQLearningAgent.get_state(env.game.getGameState())

        # Select action in state "state"
        action = monte_carlo_q_agent.compute_action_from_q_values(state)

        if action is None:
            raise IllegalActionException("Illegal action occurred.")

        """
        After choosing action, get reward.
        PLE environment method act() returns the reward that the agent has accumulated while performing the action.
        """
        reward = env.act(env.getActionSet()[action])
        training_score += reward

        max_training_score = max(training_score, max_training_score)

        game_over = env.game_over()

        # observe the result
        if state_action_reward:
            monte_carlo_q_agent.update(state_action_reward[0], state_action_reward[1], state, state_action_reward[2])

        state_action_reward = (state, action, reward)

        if game_over:
            print("===========================")
            print("Episode: " + str(episode_number))
            print("Training score: " + str(training_score))
            print("Max. training score: " + str(max_training_score))
            print("===========================\n")
            if every_100th == 100:
                results.append((episode_number, training_score))
                every_100th = 0
            episode_number += 1
            number_of_episodes -= 1
            training_score = 0
            env.reset_game()
"""
    f = open("basicq_150000.txt", "w")
    f.write(str(monte_carlo_q_agent.Q_matrix))
    f.close()

    f = open("results_150000.txt", "w")
    f.write(str(results))
    f.close()
"""

train_agent(150000)
