python train_dt2.py --track=True --project=egb-atari-1 --name=dtgpt-4_{seed} --device=cuda --seed=0 --frame-stack=4 --obj=ext --env-id=None --n-envs=64 --n-steps=256 --batch-size=1024 --ctx-len=4 --n-iters=10000 --freq-collect=32 --lr=6e-4 --lr-min=6e-5 --lr-schedule=True --ent-coef=0.0 --max-grad-norm=1.0
python train_dt2.py --track=True --project=egb-atari-1 --name=dtgpt-4_{seed} --device=cuda --seed=1 --frame-stack=4 --obj=ext --env-id=None --n-envs=64 --n-steps=256 --batch-size=1024 --ctx-len=4 --n-iters=10000 --freq-collect=32 --lr=6e-4 --lr-min=6e-5 --lr-schedule=True --ent-coef=0.0 --max-grad-norm=1.0
python train_dt2.py --track=True --project=egb-atari-1 --name=dtgpt-4_{seed} --device=cuda --seed=2 --frame-stack=4 --obj=ext --env-id=None --n-envs=64 --n-steps=256 --batch-size=1024 --ctx-len=4 --n-iters=10000 --freq-collect=32 --lr=6e-4 --lr-min=6e-5 --lr-schedule=True --ent-coef=0.0 --max-grad-norm=1.0