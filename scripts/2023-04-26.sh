
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj ext --distribution-mode hard --lstm-type residual --actions ordinal --seed 0 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj ext --distribution-mode hard --lstm-type residual --actions ordinal --seed 1 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj ext --distribution-mode hard --lstm-type residual --actions ordinal --seed 2 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj ext --distribution-mode hard --lstm-type residual --actions ordinal --seed 3 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj ext --distribution-mode hard --lstm-type residual --actions ordinal --seed 4 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj ext --distribution-mode hard --lstm-type residual --actions ordinal --seed 5 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj ext --distribution-mode hard --lstm-type residual --actions ordinal --seed 6 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj ext --distribution-mode hard --lstm-type residual --actions ordinal --seed 7 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj ext --distribution-mode hard --lstm-type residual --actions ordinal --seed 8 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj ext --distribution-mode hard --lstm-type residual --actions ordinal --seed 9 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}

python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj cov --distribution-mode hard --lstm-type residual --actions ordinal --seed 0 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj cov --distribution-mode hard --lstm-type residual --actions ordinal --seed 1 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj cov --distribution-mode hard --lstm-type residual --actions ordinal --seed 2 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj cov --distribution-mode hard --lstm-type residual --actions ordinal --seed 3 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj cov --distribution-mode hard --lstm-type residual --actions ordinal --seed 4 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj cov --distribution-mode hard --lstm-type residual --actions ordinal --seed 5 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj cov --distribution-mode hard --lstm-type residual --actions ordinal --seed 6 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj cov --distribution-mode hard --lstm-type residual --actions ordinal --seed 7 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj cov --distribution-mode hard --lstm-type residual --actions ordinal --seed 8 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj cov --distribution-mode hard --lstm-type residual --actions ordinal --seed 9 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}

python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 0 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 1 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 2 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 3 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 4 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 5 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 6 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 7 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 8 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 9 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}

python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 0 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 1 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 2 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 3 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 4 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 5 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 6 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 7 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 8 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 9 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}

python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 0 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 1 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 2 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 3 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 4 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 5 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 6 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 7 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 8 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_idmf --distribution-mode hard --lstm-type residual --actions ordinal --seed 9 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}

python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 0 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 1 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 2 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 3 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 4 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 5 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 6 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 7 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 8 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --num-levels 1e1 --obj nov_xy --distribution-mode hard --lstm-type residual --actions ordinal --seed 9 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-5 --save-agent data/{project}/{name}