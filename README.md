# Uncertainty-Driven-Replay-Memory
This repository contains code associated with the paper ***'Uncertainty-Driven Replay Memory for Reinforcement Learning'***

***environment.yml*** in this folder can be used to create a virtual environment to run the scripts.

To train the UDRM on Breakout for 10,000 timesteps, run the following command from the ***UDRM/Env_Atari/*** folder.

***python atari_train.py***

The script will run for Asterix for 2.5M timesteps for 1 seed, by default. Recalibration intervals are set at 500 for alpha and 5000 for beta. These default values can be modified by the following arguments to the above run command: <br>
***--total_timesteps*** => number of timesteps <br>
***--game_idx*** => choose a game (0-Asterix, 1-Breakout, 2-Freeway, 3-Seaquest, 4-SpaceInvaders) <br>
***--n_seeds*** => number of seeds <br>
***--alpha*** => alpha recalibration interval <br>
***--beta*** => beta recalibration interval <br>

Results will be generated and stored in the ***UDRM/results/*** folder. This folder will contain all the plots of the agent's performance and the trained network's weights.

To train UDRM on CartPole, run: <br> ***python classic_train.py*** from the ***UDRM/Env_Classic_Control/*** folder.

To train UDRM on FrozenLake, run: <br> ***python toy_text_train.py*** from the ***UDRM/Env_Toy_Text/*** folder.

The script will run for CartPole for 50000 timesteps for 10 seeds, by default. Recalibration intervals are set at 500 for alpha and 5000 for beta. These default values can be modified as mentioned above. Results will be generated and stored in the ***UDRM/results/*** folder. 