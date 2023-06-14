python train_dt2.py --track=True --project=egb-atari --name=generalist_32_{obj}_{seed} --device=cuda --seed=0 --log-video=True --env-ids Amidar Assault Asterix Atlantis BattleZone BeamRider Boxing Breakout Carnival Centipede ChopperCommand CrazyClimber DemonAttack DoubleDunk Enduro FishingDerby Freeway Frostbite Gopher Gravitar Hero IceHockey Jamesbond Kangaroo Krull KungFuMaster NameThisGame Phoenix Pooyan Qbert Riverraid RoadRunner --env-ids-test Amidar Assault Asterix Atlantis BattleZone BeamRider Boxing Breakout Carnival Centipede ChopperCommand CrazyClimber DemonAttack DoubleDunk Enduro FishingDerby Freeway Frostbite Gopher Gravitar Hero IceHockey Jamesbond Kangaroo Krull KungFuMaster NameThisGame Phoenix Pooyan Qbert Riverraid RoadRunner --obj=ext --ctx-len=4 --total-steps=1000e6 --n-envs=256 --n-steps=256 --batch-size=1024 --n-updates=64 --lr=6e-4 --lr-schedule=True --gamma=0.99 --ent-coef=0.0 --max-grad-norm=1.0 --arch=gpt --expert-agent=./data/egb-atari/specialist_{env_id}_{obj}_{seed}/agent_9.pt --load-agent=None --save-agent=./data/{project}/{name}/agent.pt