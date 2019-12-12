# CS236-Project

This repo is forked from the following repo: https://github.com/microsoft/IBAC-SNI. 

Our implementation of the autoregressive normalizing flow transition model is in torch_rl/flow_network.py. 
Our implementation of the VAE and Gaussian transition model is in torch_rl/model.py.

The scripts used to run the baseline and Gaussian transition model experiments experiments on the deterministic and stochastic versions of the Multi-Room environment are located in torch_rl/run_deterministic_4_channels.sh and torch_rl/run_stochastic_4_channels.sh.

Below is the command used to train an RL agent with a normalizing flow based transition model:
python -m scripts.train --frames 1500000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 3 --num_train_environments 100 --latent-transition-coef 0.01 --flow --num_latent_channels 4 --model 4_flows_0.01 --n_flows 4
