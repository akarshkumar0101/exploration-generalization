
python train.py --level L[0:100:1] --train-obj ext --gamma 0.999 --ent-coef 1e-1 --total-timesteps 4e7 --device cuda --track --project exploration-distillation-data --name {env}_{level:05d}_{train_obj}
python train.py --level L[0:100:1] --train-obj int --gamma 0.999 --ent-coef 1e-1 --total-timesteps 4e7 --device cuda --track --project exploration-distillation-data --name {env}_{level:05d}_{train_obj}
python train.py --level L[0:100:1] --train-obj eps --gamma 0.900 --ent-coef 1e-1 --total-timesteps 4e7 --device cuda --track --project exploration-distillation-data --name {env}_{level:05d}_{train_obj}
