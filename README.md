# Backgammon AI
This repository's goal is to teach an agent to play backgammon game. This project has already been developed, however I want to share my code with everyone to commit this repository.

## Scope

The project's scope is to: 
- create the game environment and also the agent in python.
- Optimize hyperparameters to find best agent.
- Create a simple interactive game environment that real peaple can play with the best trained agent.

## Methodology

- This was my first reinforcement project. I am learning by doing. 
- To understand the machine learning part better, initially I did not concentrate on implementing the full rules of the game. I have simplified the rules and set the winner when the first player collects all checkers to its home area. I did not implement the part where player starts to collect its checkers.
- I have implemented the [wandb](https://www.wandb.com) api to measure the and understand the learning process.
- I have kept the game board visual all in terminal view. (very simple)

## Conclusion

- The project dashboard is [here](https://app.wandb.ai/hakanonal/tavla2/sweeps/22zy2fld)
- Although I have maximized the reward among all possible parameters, my best agent still can only select about ~%18 percent times valid moves. Most of its choices are invalid moves. This performance is not usable.
- To move forward and go deeper to increase performance I have realized that I need to fully implement the game rules to the environment so that my time spent on improving the agent performance was worth. However, I have also stucked to fully implement the game rules. 
- So, It would be much apricated if you want to contribute in any ways by creating an issue and committing some code.
