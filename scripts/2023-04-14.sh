
python ppo_procgen_lstm.py --env-id heist  --total-timesteps 200e6 --learning-rate 5e-4 --idm-lr 5e-4 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e6 --obj ext --distribution-mode hard --lstm-type residual --seed 0 --device cuda --track --name {lstm_type}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-3 --save-agent data/{project}/{name}