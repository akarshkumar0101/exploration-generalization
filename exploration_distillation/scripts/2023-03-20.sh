
python train_multi.py --level-start 0 --n-levels 000001 --train-obj ext --gamma 0.999 --ent-coef 1e-1 --total-timesteps 1e7 --device cuda --track --project ed-multi --name {env}_{level:05d}_{train_obj}
python train_multi.py --level-start 0 --n-levels 000010 --train-obj ext --gamma 0.999 --ent-coef 1e-1 --total-timesteps 1e7 --device cuda --track --project ed-multi --name {env}_{level:05d}_{train_obj}
python train_multi.py --level-start 0 --n-levels 000100 --train-obj ext --gamma 0.999 --ent-coef 1e-1 --total-timesteps 1e7 --device cuda --track --project ed-multi --name {env}_{level:05d}_{train_obj}
python train_multi.py --level-start 0 --n-levels 001000 --train-obj ext --gamma 0.999 --ent-coef 1e-1 --total-timesteps 1e7 --device cuda --track --project ed-multi --name {env}_{level:05d}_{train_obj}
python train_multi.py --level-start 0 --n-levels 010000 --train-obj ext --gamma 0.999 --ent-coef 1e-1 --total-timesteps 1e7 --device cuda --track --project ed-multi --name {env}_{level:05d}_{train_obj}
python train_multi.py --level-start 0 --n-levels 100000 --train-obj ext --gamma 0.999 --ent-coef 1e-1 --total-timesteps 1e7 --device cuda --track --project ed-multi --name {env}_{level:05d}_{train_obj}

python train_multi.py --level-start 0 --n-levels 000001 --train-obj eps --gamma 0.900 --ent-coef 1e-1 --total-timesteps 1e7 --device cuda --track --project ed-multi --name {env}_{level:05d}_{train_obj}
python train_multi.py --level-start 0 --n-levels 000010 --train-obj eps --gamma 0.900 --ent-coef 1e-1 --total-timesteps 1e7 --device cuda --track --project ed-multi --name {env}_{level:05d}_{train_obj}
python train_multi.py --level-start 0 --n-levels 000100 --train-obj eps --gamma 0.900 --ent-coef 1e-1 --total-timesteps 1e7 --device cuda --track --project ed-multi --name {env}_{level:05d}_{train_obj}
python train_multi.py --level-start 0 --n-levels 001000 --train-obj eps --gamma 0.900 --ent-coef 1e-1 --total-timesteps 1e7 --device cuda --track --project ed-multi --name {env}_{level:05d}_{train_obj}
python train_multi.py --level-start 0 --n-levels 010000 --train-obj eps --gamma 0.900 --ent-coef 1e-1 --total-timesteps 1e7 --device cuda --track --project ed-multi --name {env}_{level:05d}_{train_obj}
python train_multi.py --level-start 0 --n-levels 100000 --train-obj eps --gamma 0.900 --ent-coef 1e-1 --total-timesteps 1e7 --device cuda --track --project ed-multi --name {env}_{level:05d}_{train_obj}
