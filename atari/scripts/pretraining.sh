python train.py --track=True --project=egb-atari --name=generalist_30_{pre_obj}_history_{teacher_last_agent_only}_{seed} --log-video=False --log-hist=False --device=cuda --seed=0 --env-ids Asteroids BankHeist BattleZone BeamRider Berzerk Breakout Centipede ChopperCommand CrazyClimber Freeway Frostbite Gopher Gravitar Hero Kangaroo Krull KungFuMaster MontezumaRevenge Pitfall PrivateEye Qbert RoadRunner Robotank Seaquest Solaris TimePilot Tutankham Venture YarsRevenge Zaxxon --obj=ext --total-steps=100e6 --n-envs=240 --n-steps=128 --batch-size=3840 --n-updates=16 --model=gpt --ctx-len=4 --save-agent=./data/{project}/{name}/ --full-action-space=True --lr=0.00025 --lr-warmup=True --lr-decay=none --max-grad-norm=1.0 --episodic-life=False --norm-rew=True --gamma=0.99 --gae-lambda=0.95 --norm-adv=True --ent-coef=0.001 --clip-coef=0.1 --vf-coef=0.5 --pre-obj=ext --train-klbc=True --model-teacher=cnn --ctx-len-teacher=4 --load-agent-teacher=./data/{project}/specialist_{{env_id}}_{pre_obj}_0/ --teacher-last-agent-only=True  --freq-teacher-switch=1e6 --n-steps-rnd-init=0
python train.py --track=True --project=egb-atari --name=generalist_30_{pre_obj}_history_{teacher_last_agent_only}_{seed} --log-video=False --log-hist=False --device=cuda --seed=0 --env-ids Asteroids BankHeist BattleZone BeamRider Berzerk Breakout Centipede ChopperCommand CrazyClimber Freeway Frostbite Gopher Gravitar Hero Kangaroo Krull KungFuMaster MontezumaRevenge Pitfall PrivateEye Qbert RoadRunner Robotank Seaquest Solaris TimePilot Tutankham Venture YarsRevenge Zaxxon --obj=ext --total-steps=100e6 --n-envs=240 --n-steps=128 --batch-size=3840 --n-updates=16 --model=gpt --ctx-len=4 --save-agent=./data/{project}/{name}/ --full-action-space=True --lr=0.00025 --lr-warmup=True --lr-decay=none --max-grad-norm=1.0 --episodic-life=False --norm-rew=True --gamma=0.99 --gae-lambda=0.95 --norm-adv=True --ent-coef=0.001 --clip-coef=0.1 --vf-coef=0.5 --pre-obj=ext --train-klbc=True --model-teacher=cnn --ctx-len-teacher=4 --load-agent-teacher=./data/{project}/specialist_{{env_id}}_{pre_obj}_0/ --teacher-last-agent-only=False --freq-teacher-switch=1e6 --n-steps-rnd-init=0
python train.py --track=True --project=egb-atari --name=generalist_30_{pre_obj}_history_{teacher_last_agent_only}_{seed} --log-video=False --log-hist=False --device=cuda --seed=0 --env-ids Asteroids BankHeist BattleZone BeamRider Berzerk Breakout Centipede ChopperCommand CrazyClimber Freeway Frostbite Gopher Gravitar Hero Kangaroo Krull KungFuMaster MontezumaRevenge Pitfall PrivateEye Qbert RoadRunner Robotank Seaquest Solaris TimePilot Tutankham Venture YarsRevenge Zaxxon --obj=ext --total-steps=100e6 --n-envs=240 --n-steps=128 --batch-size=3840 --n-updates=16 --model=gpt --ctx-len=4 --save-agent=./data/{project}/{name}/ --full-action-space=True --lr=0.00025 --lr-warmup=True --lr-decay=none --max-grad-norm=1.0 --episodic-life=False --norm-rew=True --gamma=0.99 --gae-lambda=0.95 --norm-adv=True --ent-coef=0.001 --clip-coef=0.1 --vf-coef=0.5 --pre-obj=rnd --train-klbc=True --model-teacher=cnn --ctx-len-teacher=4 --load-agent-teacher=./data/{project}/specialist_{{env_id}}_{pre_obj}_0/ --teacher-last-agent-only=True  --freq-teacher-switch=1e6 --n-steps-rnd-init=0
python train.py --track=True --project=egb-atari --name=generalist_30_{pre_obj}_history_{teacher_last_agent_only}_{seed} --log-video=False --log-hist=False --device=cuda --seed=0 --env-ids Asteroids BankHeist BattleZone BeamRider Berzerk Breakout Centipede ChopperCommand CrazyClimber Freeway Frostbite Gopher Gravitar Hero Kangaroo Krull KungFuMaster MontezumaRevenge Pitfall PrivateEye Qbert RoadRunner Robotank Seaquest Solaris TimePilot Tutankham Venture YarsRevenge Zaxxon --obj=ext --total-steps=100e6 --n-envs=240 --n-steps=128 --batch-size=3840 --n-updates=16 --model=gpt --ctx-len=4 --save-agent=./data/{project}/{name}/ --full-action-space=True --lr=0.00025 --lr-warmup=True --lr-decay=none --max-grad-norm=1.0 --episodic-life=False --norm-rew=True --gamma=0.99 --gae-lambda=0.95 --norm-adv=True --ent-coef=0.001 --clip-coef=0.1 --vf-coef=0.5 --pre-obj=rnd --train-klbc=True --model-teacher=cnn --ctx-len-teacher=4 --load-agent-teacher=./data/{project}/specialist_{{env_id}}_{pre_obj}_0/ --teacher-last-agent-only=False --freq-teacher-switch=1e6 --n-steps-rnd-init=0
python train.py --track=True --project=egb-atari --name=generalist_30_{pre_obj}_history_{teacher_last_agent_only}_{seed} --log-video=False --log-hist=False --device=cuda --seed=1 --env-ids Asteroids BankHeist BattleZone BeamRider Berzerk Breakout Centipede ChopperCommand CrazyClimber Freeway Frostbite Gopher Gravitar Hero Kangaroo Krull KungFuMaster MontezumaRevenge Pitfall PrivateEye Qbert RoadRunner Robotank Seaquest Solaris TimePilot Tutankham Venture YarsRevenge Zaxxon --obj=ext --total-steps=100e6 --n-envs=240 --n-steps=128 --batch-size=3840 --n-updates=16 --model=gpt --ctx-len=4 --save-agent=./data/{project}/{name}/ --full-action-space=True --lr=0.00025 --lr-warmup=True --lr-decay=none --max-grad-norm=1.0 --episodic-life=False --norm-rew=True --gamma=0.99 --gae-lambda=0.95 --norm-adv=True --ent-coef=0.001 --clip-coef=0.1 --vf-coef=0.5 --pre-obj=ext --train-klbc=True --model-teacher=cnn --ctx-len-teacher=4 --load-agent-teacher=./data/{project}/specialist_{{env_id}}_{pre_obj}_0/ --teacher-last-agent-only=True  --freq-teacher-switch=1e6 --n-steps-rnd-init=0
python train.py --track=True --project=egb-atari --name=generalist_30_{pre_obj}_history_{teacher_last_agent_only}_{seed} --log-video=False --log-hist=False --device=cuda --seed=1 --env-ids Asteroids BankHeist BattleZone BeamRider Berzerk Breakout Centipede ChopperCommand CrazyClimber Freeway Frostbite Gopher Gravitar Hero Kangaroo Krull KungFuMaster MontezumaRevenge Pitfall PrivateEye Qbert RoadRunner Robotank Seaquest Solaris TimePilot Tutankham Venture YarsRevenge Zaxxon --obj=ext --total-steps=100e6 --n-envs=240 --n-steps=128 --batch-size=3840 --n-updates=16 --model=gpt --ctx-len=4 --save-agent=./data/{project}/{name}/ --full-action-space=True --lr=0.00025 --lr-warmup=True --lr-decay=none --max-grad-norm=1.0 --episodic-life=False --norm-rew=True --gamma=0.99 --gae-lambda=0.95 --norm-adv=True --ent-coef=0.001 --clip-coef=0.1 --vf-coef=0.5 --pre-obj=ext --train-klbc=True --model-teacher=cnn --ctx-len-teacher=4 --load-agent-teacher=./data/{project}/specialist_{{env_id}}_{pre_obj}_0/ --teacher-last-agent-only=False --freq-teacher-switch=1e6 --n-steps-rnd-init=0
python train.py --track=True --project=egb-atari --name=generalist_30_{pre_obj}_history_{teacher_last_agent_only}_{seed} --log-video=False --log-hist=False --device=cuda --seed=1 --env-ids Asteroids BankHeist BattleZone BeamRider Berzerk Breakout Centipede ChopperCommand CrazyClimber Freeway Frostbite Gopher Gravitar Hero Kangaroo Krull KungFuMaster MontezumaRevenge Pitfall PrivateEye Qbert RoadRunner Robotank Seaquest Solaris TimePilot Tutankham Venture YarsRevenge Zaxxon --obj=ext --total-steps=100e6 --n-envs=240 --n-steps=128 --batch-size=3840 --n-updates=16 --model=gpt --ctx-len=4 --save-agent=./data/{project}/{name}/ --full-action-space=True --lr=0.00025 --lr-warmup=True --lr-decay=none --max-grad-norm=1.0 --episodic-life=False --norm-rew=True --gamma=0.99 --gae-lambda=0.95 --norm-adv=True --ent-coef=0.001 --clip-coef=0.1 --vf-coef=0.5 --pre-obj=rnd --train-klbc=True --model-teacher=cnn --ctx-len-teacher=4 --load-agent-teacher=./data/{project}/specialist_{{env_id}}_{pre_obj}_0/ --teacher-last-agent-only=True  --freq-teacher-switch=1e6 --n-steps-rnd-init=0
python train.py --track=True --project=egb-atari --name=generalist_30_{pre_obj}_history_{teacher_last_agent_only}_{seed} --log-video=False --log-hist=False --device=cuda --seed=1 --env-ids Asteroids BankHeist BattleZone BeamRider Berzerk Breakout Centipede ChopperCommand CrazyClimber Freeway Frostbite Gopher Gravitar Hero Kangaroo Krull KungFuMaster MontezumaRevenge Pitfall PrivateEye Qbert RoadRunner Robotank Seaquest Solaris TimePilot Tutankham Venture YarsRevenge Zaxxon --obj=ext --total-steps=100e6 --n-envs=240 --n-steps=128 --batch-size=3840 --n-updates=16 --model=gpt --ctx-len=4 --save-agent=./data/{project}/{name}/ --full-action-space=True --lr=0.00025 --lr-warmup=True --lr-decay=none --max-grad-norm=1.0 --episodic-life=False --norm-rew=True --gamma=0.99 --gae-lambda=0.95 --norm-adv=True --ent-coef=0.001 --clip-coef=0.1 --vf-coef=0.5 --pre-obj=rnd --train-klbc=True --model-teacher=cnn --ctx-len-teacher=4 --load-agent-teacher=./data/{project}/specialist_{{env_id}}_{pre_obj}_0/ --teacher-last-agent-only=False --freq-teacher-switch=1e6 --n-steps-rnd-init=0