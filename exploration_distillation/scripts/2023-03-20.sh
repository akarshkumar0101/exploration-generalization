
python train_multi.py --level-start 0000000 --n-levels 0000001 --train-obj ext --gamma 0.999 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
python train_multi.py --level-start 0000000 --n-levels 0000010 --train-obj ext --gamma 0.999 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
python train_multi.py --level-start 0000000 --n-levels 0000100 --train-obj ext --gamma 0.999 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
python train_multi.py --level-start 0000000 --n-levels 0001000 --train-obj ext --gamma 0.999 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
python train_multi.py --level-start 0000000 --n-levels 0010000 --train-obj ext --gamma 0.999 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
python train_multi.py --level-start 0000000 --n-levels 0100000 --train-obj ext --gamma 0.999 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
python train_multi.py --level-start 0000000 --n-levels 1000000 --train-obj ext --gamma 0.999 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}

python train_multi.py --level-start 0000000 --n-levels 0000001 --train-obj eps --gamma 0.900 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
python train_multi.py --level-start 0000000 --n-levels 0000010 --train-obj eps --gamma 0.900 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
python train_multi.py --level-start 0000000 --n-levels 0000100 --train-obj eps --gamma 0.900 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
python train_multi.py --level-start 0000000 --n-levels 0001000 --train-obj eps --gamma 0.900 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
python train_multi.py --level-start 0000000 --n-levels 0010000 --train-obj eps --gamma 0.900 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
python train_multi.py --level-start 0000000 --n-levels 0100000 --train-obj eps --gamma 0.900 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
python train_multi.py --level-start 0000000 --n-levels 1000000 --train-obj eps --gamma 0.900 --ent-coef 1e-2 --total-timesteps 50e6 --device cuda --track --project ed-multi --name {env}_{n_levels:08d}_{train_obj}
