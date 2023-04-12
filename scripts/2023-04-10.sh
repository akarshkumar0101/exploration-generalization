

python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 1e-3 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e3 --obj ext --distribution-mode easy --lstm lstm          --seed 0 --device cuda --track --name {lstm}_{env_id}_{num_levels:06.0f}_{obj}_{seed}              --project egb-lstm-2 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 1e-3 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e3 --obj ext --distribution-mode easy --lstm ignore-lstm   --seed 0 --device cuda --track --name {lstm}_{env_id}_{num_levels:06.0f}_{obj}_{seed}   --project egb-lstm-2 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 1e-3 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e3 --obj ext --distribution-mode easy --lstm no-recurrence --seed 0 --device cuda --track --name {lstm}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-2 --save-agent data/{project}/{name}

python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 1e-3 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e3 --obj ext --distribution-mode easy --lstm lstm          --seed 1 --device cuda --track --name {lstm}_{env_id}_{num_levels:06.0f}_{obj}_{seed}              --project egb-lstm-2 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 1e-3 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e3 --obj ext --distribution-mode easy --lstm ignore-lstm   --seed 1 --device cuda --track --name {lstm}_{env_id}_{num_levels:06.0f}_{obj}_{seed}   --project egb-lstm-2 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 1e-3 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e3 --obj ext --distribution-mode easy --lstm no-recurrence --seed 1 --device cuda --track --name {lstm}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-2 --save-agent data/{project}/{name}

python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 1e-3 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e3 --obj ext --distribution-mode easy --lstm lstm          --seed 2 --device cuda --track --name {lstm}_{env_id}_{num_levels:06.0f}_{obj}_{seed}              --project egb-lstm-2 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 1e-3 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e3 --obj ext --distribution-mode easy --lstm ignore-lstm   --seed 2 --device cuda --track --name {lstm}_{env_id}_{num_levels:06.0f}_{obj}_{seed}   --project egb-lstm-2 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 1e-3 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e3 --obj ext --distribution-mode easy --lstm no-recurrence --seed 2 --device cuda --track --name {lstm}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-2 --save-agent data/{project}/{name}

python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 1e-3 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e3 --obj ext --distribution-mode easy --lstm lstm          --seed 3 --device cuda --track --name {lstm}_{env_id}_{num_levels:06.0f}_{obj}_{seed}              --project egb-lstm-2 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 1e-3 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e3 --obj ext --distribution-mode easy --lstm ignore-lstm   --seed 3 --device cuda --track --name {lstm}_{env_id}_{num_levels:06.0f}_{obj}_{seed}   --project egb-lstm-2 --save-agent data/{project}/{name}
python ppo_procgen_lstm.py --env-id miner --total-timesteps 25e6 --learning-rate 5e-4 --idm-lr 1e-3 --num-envs 64 --num-steps 256 --num-minibatches 8 --update-epochs 3 --ent-coef 1e-2 --start-level 0 --num-levels 1e3 --obj ext --distribution-mode easy --lstm no-recurrence --seed 3 --device cuda --track --name {lstm}_{env_id}_{num_levels:06.0f}_{obj}_{seed} --project egb-lstm-2 --save-agent data/{project}/{name}