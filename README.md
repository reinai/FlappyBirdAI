# FlappyBirdAI

## About project
Implementation of Flappy Bird reinforcement learning agent by using two approaches:
  * Q learning (2000 episodes - trained about 2 minutes, 150 000 episodes - trained 7 hours)
  * Monte Carlo Q learning (2000 episodes - trained about 9 minutes, 50 000 episodes - don't know the exact time, 150 000 episodes - trained 11.5 hours)
  * Deep reinforcement learning (announced)
  
  In folder [basic-q-learning](https://github.com/reinai/FlappyBirdAI/tree/master/FlappyBird/basic-q-learning) we trained our model with basicq_2000.txt and basicq_150000.txt, so just run runBasicQLearning.py and open one of these files, and in folder [monte-carlo-q-learning](https://github.com/reinai/FlappyBirdAI/tree/master/FlappyBird/monte-carlo-q-learning) with monte_2000.txt and results_150000.txt and run runMonteCarloQLearning.py.

## Programs & libraries needed in order to run this project 
* [PLE: A Reinforcement Learning Environment](https://pygame-learning-environment.readthedocs.io/en/latest/#) mimicking the Arcade Learning Environment interface, allowing a quick start to Reinforcement Learning in Python
* [NumPy](https://www.numpy.org/) fundamental package for scientific computing with Python
* [Pillow](https://python-pillow.org/) Python Imaging Library
* [pyGame](https://www.pygame.org/news) cross-platform set of Python modules designed for writing video games
* other helper modules which can be installed manually

## Game run example
![Preview](https://github.com/reinai/FlappyBirdAI/blob/master/FlappyBird/ezgif-5-809f5b1ff0d3.gif)

## Results after 2000 episodes
![alt text](https://github.com/reinai/FlappyBirdAI/blob/master/FlappyBird/results/2000%20iterations/basic_q.PNG)
Basic Q learning has still problems after 2000 episodes, sometimes getting score about 4. Training lasted about 2 minutes.
![alt text](https://github.com/reinai/FlappyBirdAI/blob/master/FlappyBird/results/2000%20iterations/monte_carlo_q.PNG)
Monte Carlo Q learning after 2000 episodes in which training lasted about 9 minutes has tremendously better results, averaging at about score of 80.

## Q learning and Monte Carlo Q learning
![alt text](https://github.com/reinai/FlappyBirdAI/blob/master/FlappyBird/poster_image.PNG)
