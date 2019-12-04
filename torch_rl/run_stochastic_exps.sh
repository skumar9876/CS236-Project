#!/bin/bash
python -m scripts.train --frames 150000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --model stochastic/ppo_1 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 1 --num_train_environments 100 --stochastic >/dev/null &
python -m scripts.train --frames 150000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --model stochastic/ppo_2 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 2 --num_train_environments 100 --stochastic >/dev/null &
python -m scripts.train --frames 150000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --model stochastic/ppo_3 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 3 --num_train_environments 100 --stochastic >/dev/null &
python -m scripts.train --frames 150000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --model stochastic/ppo_vae_1 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 1 --num_train_environments 100  --KLD-coef 0.05 --reconstruction-likelihood-coef 0.05 --stochastic >/dev/null &
python -m scripts.train --frames 150000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --model stochastic/ppo_vae_2 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 2 --num_train_environments 100  --KLD-coef 0.05 --reconstruction-likelihood-coef 0.05 --stochastic >/dev/null &
python -m scripts.train --frames 150000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --model stochastic/ppo_vae_3 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 3 --num_train_environments 100  --KLD-coef 0.05 --reconstruction-likelihood-coef 0.05 --stochastic >/dev/null & 
python -m scripts.train --frames 150000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --model stochastic/ppo_vae_transition_1 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 1 --num_train_environments 100  --KLD-coef 0.05 --reconstruction-likelihood-coef 0.05 --latent-transition-coef 0.1 --stochastic >/dev/null &
python -m scripts.train --frames 150000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --model stochastic/ppo_vae_transition_2 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 2 --num_train_environments 100  --KLD-coef 0.05 --reconstruction-likelihood-coef 0.05 --latent-transition-coef 0.1 --stochastic >/dev/null &
python -m scripts.train --frames 150000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --model stochastic/ppo_vae_transition_3 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 3 --num_train_environments 100  --KLD-coef 0.05 --reconstruction-likelihood-coef 0.05 --latent-transition-coef 0.1 --stochastic >/dev/null &
python -m scripts.train --frames 150000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --model stochastic/ppo_transition_1 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 1 --num_train_environments 100  --latent-transition-coef 0.1 --stochastic >/dev/null &
python -m scripts.train --frames 150000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --model stochastic/ppo_transition_2 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 2 --num_train_environments 100  --latent-transition-coef 0.1 --stochastic >/dev/null &
python -m scripts.train --frames 150000 --algo ppo --env MiniGrid-MultiRoom-N5r_1_2r-v1 --model stochastic/ppo_transition_3 --save-interval 10 --eval-interval 0 --tb --model_type default2 --seed 3 --num_train_environments 100  --latent-transition-coef 0.1 --stochastic >/dev/null &