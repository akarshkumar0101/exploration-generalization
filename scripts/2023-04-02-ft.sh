
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100000 --seed 0 --ent-coef 0.0 --kl0-coef 1e-2 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 0 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100001 --seed 1 --ent-coef 0.0 --kl0-coef 1e-2 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 1 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100002 --seed 2 --ent-coef 0.0 --kl0-coef 1e-2 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 2 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100003 --seed 3 --ent-coef 0.0 --kl0-coef 1e-2 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 3 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100004 --seed 4 --ent-coef 0.0 --kl0-coef 1e-2 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 4 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}

python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100000 --seed 0 --ent-coef 0.0 --kl0-coef 1e-1 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 0 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100001 --seed 1 --ent-coef 0.0 --kl0-coef 1e-1 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 1 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100002 --seed 2 --ent-coef 0.0 --kl0-coef 1e-1 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 2 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100003 --seed 3 --ent-coef 0.0 --kl0-coef 1e-1 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 3 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100004 --seed 4 --ent-coef 0.0 --kl0-coef 1e-1 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 4 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}

python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100000 --seed 0 --ent-coef 0.0 --kl0-coef 1e-0 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 0 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100001 --seed 1 --ent-coef 0.0 --kl0-coef 1e-0 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 1 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100002 --seed 2 --ent-coef 0.0 --kl0-coef 1e-0 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 2 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100003 --seed 3 --ent-coef 0.0 --kl0-coef 1e-0 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 3 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
python ppo_procgen.py --distribution-mode easy --total-timesteps 25e6 --num-levels 1e0 --warmup-critic-steps 1e6 --save-agent data_ft/{name} --device cuda --track --name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed} --project egb3-ft --obj ext --start-level 100004 --seed 4 --ent-coef 0.0 --kl0-coef 1e-0 --pre-env-id miner --pre-num-levels 1e1 --pre-obj eps --pre-seed 4 --load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}


#python ppo_procgen.py
#--distribution-mode easy
#--total-timesteps 25e6
#--num-levels 1e0
#--warmup-critic-steps 1e6
#--save-agent data_ft/{name}
#--device cuda
#--track
#--name ft_{env_id}_{num_levels:06.0f}_{obj}_{seed}_kl{kl0_coef:05.0e}_pre_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
#--project egb3-ft
#
#--obj ext
#--start-level 100000
#--seed 0
#--ent-coef 0.0
#--kl0-coef 1e-1
#
#--pre-env-id miner
#--pre-num-levels 1e1
#--pre-obj eps
#--pre-seed 0
#--load-agent data/{pre_env_id}_{pre_num_levels:06.0f}_{pre_obj}_{pre_seed}
