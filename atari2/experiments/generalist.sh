python bc.py --track=True --project="egb_generalist" --name="final_ckpt_0_0000" --device="cuda" --model="trans_64" --save_ckpt="./data/egb_generalist/final_ckpt_0_0000/ckpt_{i_iter}.pt" --env_ids Amidar Assault Asteroids Atlantis BankHeist BattleZone Bowling Boxing Breakout Centipede ChopperCommand CrazyClimber Defender DemonAttack DoubleDunk Enduro FishingDerby Freeway Frostbite Gopher Gravitar Hero IceHockey Kangaroo Krull KungFuMaster MontezumaRevenge MsPacman NameThisGame Phoenix Pitfall Pong PrivateEye Riverraid RoadRunner Robotank Seaquest Solaris SpaceInvaders StarGunner Surround Tennis TimePilot Tutankham UpNDown Venture VideoPinball YarsRevenge --n_iters=1000 --load_ckpt_teacher ./data/egb_specialist/Amidar_0000/ckpt_9999.pt ./data/egb_specialist/Assault_0000/ckpt_9999.pt ./data/egb_specialist/Asteroids_0000/ckpt_9999.pt ./data/egb_specialist/Atlantis_0000/ckpt_9999.pt ./data/egb_specialist/BankHeist_0000/ckpt_9999.pt ./data/egb_specialist/BattleZone_0000/ckpt_9999.pt ./data/egb_specialist/Bowling_0000/ckpt_9999.pt ./data/egb_specialist/Boxing_0000/ckpt_9999.pt ./data/egb_specialist/Breakout_0000/ckpt_9999.pt ./data/egb_specialist/Centipede_0000/ckpt_9999.pt ./data/egb_specialist/ChopperCommand_0000/ckpt_9999.pt ./data/egb_specialist/CrazyClimber_0000/ckpt_9999.pt ./data/egb_specialist/Defender_0000/ckpt_9999.pt ./data/egb_specialist/DemonAttack_0000/ckpt_9999.pt ./data/egb_specialist/DoubleDunk_0000/ckpt_9999.pt ./data/egb_specialist/Enduro_0000/ckpt_9999.pt ./data/egb_specialist/FishingDerby_0000/ckpt_9999.pt ./data/egb_specialist/Freeway_0000/ckpt_9999.pt ./data/egb_specialist/Frostbite_0000/ckpt_9999.pt ./data/egb_specialist/Gopher_0000/ckpt_9999.pt ./data/egb_specialist/Gravitar_0000/ckpt_9999.pt ./data/egb_specialist/Hero_0000/ckpt_9999.pt ./data/egb_specialist/IceHockey_0000/ckpt_9999.pt ./data/egb_specialist/Kangaroo_0000/ckpt_9999.pt ./data/egb_specialist/Krull_0000/ckpt_9999.pt ./data/egb_specialist/KungFuMaster_0000/ckpt_9999.pt ./data/egb_specialist/MontezumaRevenge_0000/ckpt_9999.pt ./data/egb_specialist/MsPacman_0000/ckpt_9999.pt ./data/egb_specialist/NameThisGame_0000/ckpt_9999.pt ./data/egb_specialist/Phoenix_0000/ckpt_9999.pt ./data/egb_specialist/Pitfall_0000/ckpt_9999.pt ./data/egb_specialist/Pong_0000/ckpt_9999.pt ./data/egb_specialist/PrivateEye_0000/ckpt_9999.pt ./data/egb_specialist/Riverraid_0000/ckpt_9999.pt ./data/egb_specialist/RoadRunner_0000/ckpt_9999.pt ./data/egb_specialist/Robotank_0000/ckpt_9999.pt ./data/egb_specialist/Seaquest_0000/ckpt_9999.pt ./data/egb_specialist/Solaris_0000/ckpt_9999.pt ./data/egb_specialist/SpaceInvaders_0000/ckpt_9999.pt ./data/egb_specialist/StarGunner_0000/ckpt_9999.pt ./data/egb_specialist/Surround_0000/ckpt_9999.pt ./data/egb_specialist/Tennis_0000/ckpt_9999.pt ./data/egb_specialist/TimePilot_0000/ckpt_9999.pt ./data/egb_specialist/Tutankham_0000/ckpt_9999.pt ./data/egb_specialist/UpNDown_0000/ckpt_9999.pt ./data/egb_specialist/Venture_0000/ckpt_9999.pt ./data/egb_specialist/VideoPinball_0000/ckpt_9999.pt ./data/egb_specialist/YarsRevenge_0000/ckpt_9999.pt --i_split=0 --strategy="final_ckpt"
python bc.py --track=True --project="egb_generalist" --name="all_ckpts_0_0000"  --device="cuda" --model="trans_64" --save_ckpt="./data/egb_generalist/all_ckpts_0_0000/ckpt_{i_iter}.pt"  --env_ids Amidar Assault Asteroids Atlantis BankHeist BattleZone Bowling Boxing Breakout Centipede ChopperCommand CrazyClimber Defender DemonAttack DoubleDunk Enduro FishingDerby Freeway Frostbite Gopher Gravitar Hero IceHockey Kangaroo Krull KungFuMaster MontezumaRevenge MsPacman NameThisGame Phoenix Pitfall Pong PrivateEye Riverraid RoadRunner Robotank Seaquest Solaris SpaceInvaders StarGunner Surround Tennis TimePilot Tutankham UpNDown Venture VideoPinball YarsRevenge --n_iters=1000 --load_ckpt_teacher ./data/egb_specialist/Amidar_0000/ckpt_*.pt ./data/egb_specialist/Assault_0000/ckpt_*.pt ./data/egb_specialist/Asteroids_0000/ckpt_*.pt ./data/egb_specialist/Atlantis_0000/ckpt_*.pt ./data/egb_specialist/BankHeist_0000/ckpt_*.pt ./data/egb_specialist/BattleZone_0000/ckpt_*.pt ./data/egb_specialist/Bowling_0000/ckpt_*.pt ./data/egb_specialist/Boxing_0000/ckpt_*.pt ./data/egb_specialist/Breakout_0000/ckpt_*.pt ./data/egb_specialist/Centipede_0000/ckpt_*.pt ./data/egb_specialist/ChopperCommand_0000/ckpt_*.pt ./data/egb_specialist/CrazyClimber_0000/ckpt_*.pt ./data/egb_specialist/Defender_0000/ckpt_*.pt ./data/egb_specialist/DemonAttack_0000/ckpt_*.pt ./data/egb_specialist/DoubleDunk_0000/ckpt_*.pt ./data/egb_specialist/Enduro_0000/ckpt_*.pt ./data/egb_specialist/FishingDerby_0000/ckpt_*.pt ./data/egb_specialist/Freeway_0000/ckpt_*.pt ./data/egb_specialist/Frostbite_0000/ckpt_*.pt ./data/egb_specialist/Gopher_0000/ckpt_*.pt ./data/egb_specialist/Gravitar_0000/ckpt_*.pt ./data/egb_specialist/Hero_0000/ckpt_*.pt ./data/egb_specialist/IceHockey_0000/ckpt_*.pt ./data/egb_specialist/Kangaroo_0000/ckpt_*.pt ./data/egb_specialist/Krull_0000/ckpt_*.pt ./data/egb_specialist/KungFuMaster_0000/ckpt_*.pt ./data/egb_specialist/MontezumaRevenge_0000/ckpt_*.pt ./data/egb_specialist/MsPacman_0000/ckpt_*.pt ./data/egb_specialist/NameThisGame_0000/ckpt_*.pt ./data/egb_specialist/Phoenix_0000/ckpt_*.pt ./data/egb_specialist/Pitfall_0000/ckpt_*.pt ./data/egb_specialist/Pong_0000/ckpt_*.pt ./data/egb_specialist/PrivateEye_0000/ckpt_*.pt ./data/egb_specialist/Riverraid_0000/ckpt_*.pt ./data/egb_specialist/RoadRunner_0000/ckpt_*.pt ./data/egb_specialist/Robotank_0000/ckpt_*.pt ./data/egb_specialist/Seaquest_0000/ckpt_*.pt ./data/egb_specialist/Solaris_0000/ckpt_*.pt ./data/egb_specialist/SpaceInvaders_0000/ckpt_*.pt ./data/egb_specialist/StarGunner_0000/ckpt_*.pt ./data/egb_specialist/Surround_0000/ckpt_*.pt ./data/egb_specialist/Tennis_0000/ckpt_*.pt ./data/egb_specialist/TimePilot_0000/ckpt_*.pt ./data/egb_specialist/Tutankham_0000/ckpt_*.pt ./data/egb_specialist/UpNDown_0000/ckpt_*.pt ./data/egb_specialist/Venture_0000/ckpt_*.pt ./data/egb_specialist/VideoPinball_0000/ckpt_*.pt ./data/egb_specialist/YarsRevenge_0000/ckpt_*.pt                                                                                                                                                 --i_split=0 --strategy="all_ckpts" 
python bc.py --track=True --project="egb_generalist" --name="final_ckpt_1_0000" --device="cuda" --model="trans_64" --save_ckpt="./data/egb_generalist/final_ckpt_1_0000/ckpt_{i_iter}.pt" --env_ids Alien Amidar Asterix BankHeist BattleZone BeamRider Berzerk Bowling Breakout Centipede ChopperCommand CrazyClimber Defender DemonAttack DoubleDunk Enduro FishingDerby Freeway Frostbite Gopher Hero IceHockey Jamesbond Kangaroo Krull KungFuMaster MontezumaRevenge MsPacman NameThisGame Phoenix Pong PrivateEye Qbert Riverraid RoadRunner Robotank Seaquest Solaris SpaceInvaders StarGunner Surround Tennis TimePilot UpNDown VideoPinball WizardOfWor YarsRevenge Zaxxon   --n_iters=1000 --load_ckpt_teacher ./data/egb_specialist/Alien_0000/ckpt_9999.pt ./data/egb_specialist/Amidar_0000/ckpt_9999.pt ./data/egb_specialist/Asterix_0000/ckpt_9999.pt ./data/egb_specialist/BankHeist_0000/ckpt_9999.pt ./data/egb_specialist/BattleZone_0000/ckpt_9999.pt ./data/egb_specialist/BeamRider_0000/ckpt_9999.pt ./data/egb_specialist/Berzerk_0000/ckpt_9999.pt ./data/egb_specialist/Bowling_0000/ckpt_9999.pt ./data/egb_specialist/Breakout_0000/ckpt_9999.pt ./data/egb_specialist/Centipede_0000/ckpt_9999.pt ./data/egb_specialist/ChopperCommand_0000/ckpt_9999.pt ./data/egb_specialist/CrazyClimber_0000/ckpt_9999.pt ./data/egb_specialist/Defender_0000/ckpt_9999.pt ./data/egb_specialist/DemonAttack_0000/ckpt_9999.pt ./data/egb_specialist/DoubleDunk_0000/ckpt_9999.pt ./data/egb_specialist/Enduro_0000/ckpt_9999.pt ./data/egb_specialist/FishingDerby_0000/ckpt_9999.pt ./data/egb_specialist/Freeway_0000/ckpt_9999.pt ./data/egb_specialist/Frostbite_0000/ckpt_9999.pt ./data/egb_specialist/Gopher_0000/ckpt_9999.pt ./data/egb_specialist/Hero_0000/ckpt_9999.pt ./data/egb_specialist/IceHockey_0000/ckpt_9999.pt ./data/egb_specialist/Jamesbond_0000/ckpt_9999.pt ./data/egb_specialist/Kangaroo_0000/ckpt_9999.pt ./data/egb_specialist/Krull_0000/ckpt_9999.pt ./data/egb_specialist/KungFuMaster_0000/ckpt_9999.pt ./data/egb_specialist/MontezumaRevenge_0000/ckpt_9999.pt ./data/egb_specialist/MsPacman_0000/ckpt_9999.pt ./data/egb_specialist/NameThisGame_0000/ckpt_9999.pt ./data/egb_specialist/Phoenix_0000/ckpt_9999.pt ./data/egb_specialist/Pong_0000/ckpt_9999.pt ./data/egb_specialist/PrivateEye_0000/ckpt_9999.pt ./data/egb_specialist/Qbert_0000/ckpt_9999.pt ./data/egb_specialist/Riverraid_0000/ckpt_9999.pt ./data/egb_specialist/RoadRunner_0000/ckpt_9999.pt ./data/egb_specialist/Robotank_0000/ckpt_9999.pt ./data/egb_specialist/Seaquest_0000/ckpt_9999.pt ./data/egb_specialist/Solaris_0000/ckpt_9999.pt ./data/egb_specialist/SpaceInvaders_0000/ckpt_9999.pt ./data/egb_specialist/StarGunner_0000/ckpt_9999.pt ./data/egb_specialist/Surround_0000/ckpt_9999.pt ./data/egb_specialist/Tennis_0000/ckpt_9999.pt ./data/egb_specialist/TimePilot_0000/ckpt_9999.pt ./data/egb_specialist/UpNDown_0000/ckpt_9999.pt ./data/egb_specialist/VideoPinball_0000/ckpt_9999.pt ./data/egb_specialist/WizardOfWor_0000/ckpt_9999.pt ./data/egb_specialist/YarsRevenge_0000/ckpt_9999.pt ./data/egb_specialist/Zaxxon_0000/ckpt_9999.pt   --i_split=1 --strategy="final_ckpt"
python bc.py --track=True --project="egb_generalist" --name="all_ckpts_1_0000"  --device="cuda" --model="trans_64" --save_ckpt="./data/egb_generalist/all_ckpts_1_0000/ckpt_{i_iter}.pt"  --env_ids Alien Amidar Asterix BankHeist BattleZone BeamRider Berzerk Bowling Breakout Centipede ChopperCommand CrazyClimber Defender DemonAttack DoubleDunk Enduro FishingDerby Freeway Frostbite Gopher Hero IceHockey Jamesbond Kangaroo Krull KungFuMaster MontezumaRevenge MsPacman NameThisGame Phoenix Pong PrivateEye Qbert Riverraid RoadRunner Robotank Seaquest Solaris SpaceInvaders StarGunner Surround Tennis TimePilot UpNDown VideoPinball WizardOfWor YarsRevenge Zaxxon   --n_iters=1000 --load_ckpt_teacher ./data/egb_specialist/Alien_0000/ckpt_*.pt ./data/egb_specialist/Amidar_0000/ckpt_*.pt ./data/egb_specialist/Asterix_0000/ckpt_*.pt ./data/egb_specialist/BankHeist_0000/ckpt_*.pt ./data/egb_specialist/BattleZone_0000/ckpt_*.pt ./data/egb_specialist/BeamRider_0000/ckpt_*.pt ./data/egb_specialist/Berzerk_0000/ckpt_*.pt ./data/egb_specialist/Bowling_0000/ckpt_*.pt ./data/egb_specialist/Breakout_0000/ckpt_*.pt ./data/egb_specialist/Centipede_0000/ckpt_*.pt ./data/egb_specialist/ChopperCommand_0000/ckpt_*.pt ./data/egb_specialist/CrazyClimber_0000/ckpt_*.pt ./data/egb_specialist/Defender_0000/ckpt_*.pt ./data/egb_specialist/DemonAttack_0000/ckpt_*.pt ./data/egb_specialist/DoubleDunk_0000/ckpt_*.pt ./data/egb_specialist/Enduro_0000/ckpt_*.pt ./data/egb_specialist/FishingDerby_0000/ckpt_*.pt ./data/egb_specialist/Freeway_0000/ckpt_*.pt ./data/egb_specialist/Frostbite_0000/ckpt_*.pt ./data/egb_specialist/Gopher_0000/ckpt_*.pt ./data/egb_specialist/Hero_0000/ckpt_*.pt ./data/egb_specialist/IceHockey_0000/ckpt_*.pt ./data/egb_specialist/Jamesbond_0000/ckpt_*.pt ./data/egb_specialist/Kangaroo_0000/ckpt_*.pt ./data/egb_specialist/Krull_0000/ckpt_*.pt ./data/egb_specialist/KungFuMaster_0000/ckpt_*.pt ./data/egb_specialist/MontezumaRevenge_0000/ckpt_*.pt ./data/egb_specialist/MsPacman_0000/ckpt_*.pt ./data/egb_specialist/NameThisGame_0000/ckpt_*.pt ./data/egb_specialist/Phoenix_0000/ckpt_*.pt ./data/egb_specialist/Pong_0000/ckpt_*.pt ./data/egb_specialist/PrivateEye_0000/ckpt_*.pt ./data/egb_specialist/Qbert_0000/ckpt_*.pt ./data/egb_specialist/Riverraid_0000/ckpt_*.pt ./data/egb_specialist/RoadRunner_0000/ckpt_*.pt ./data/egb_specialist/Robotank_0000/ckpt_*.pt ./data/egb_specialist/Seaquest_0000/ckpt_*.pt ./data/egb_specialist/Solaris_0000/ckpt_*.pt ./data/egb_specialist/SpaceInvaders_0000/ckpt_*.pt ./data/egb_specialist/StarGunner_0000/ckpt_*.pt ./data/egb_specialist/Surround_0000/ckpt_*.pt ./data/egb_specialist/Tennis_0000/ckpt_*.pt ./data/egb_specialist/TimePilot_0000/ckpt_*.pt ./data/egb_specialist/UpNDown_0000/ckpt_*.pt ./data/egb_specialist/VideoPinball_0000/ckpt_*.pt ./data/egb_specialist/WizardOfWor_0000/ckpt_*.pt ./data/egb_specialist/YarsRevenge_0000/ckpt_*.pt ./data/egb_specialist/Zaxxon_0000/ckpt_*.pt                                                                                                                                                   --i_split=1 --strategy="all_ckpts" 
python bc.py --track=True --project="egb_generalist" --name="final_ckpt_2_0000" --device="cuda" --model="trans_64" --save_ckpt="./data/egb_generalist/final_ckpt_2_0000/ckpt_{i_iter}.pt" --env_ids Alien Amidar Assault Asterix Asteroids Atlantis BattleZone BeamRider Berzerk Bowling Boxing Breakout Centipede ChopperCommand CrazyClimber Defender DemonAttack DoubleDunk FishingDerby Freeway Gopher Gravitar Hero IceHockey Jamesbond Kangaroo Krull KungFuMaster MsPacman NameThisGame Phoenix Pitfall Pong Qbert Riverraid RoadRunner Robotank Seaquest Solaris StarGunner Surround Tennis Tutankham Venture VideoPinball WizardOfWor YarsRevenge Zaxxon                     --n_iters=1000 --load_ckpt_teacher ./data/egb_specialist/Alien_0000/ckpt_9999.pt ./data/egb_specialist/Amidar_0000/ckpt_9999.pt ./data/egb_specialist/Assault_0000/ckpt_9999.pt ./data/egb_specialist/Asterix_0000/ckpt_9999.pt ./data/egb_specialist/Asteroids_0000/ckpt_9999.pt ./data/egb_specialist/Atlantis_0000/ckpt_9999.pt ./data/egb_specialist/BattleZone_0000/ckpt_9999.pt ./data/egb_specialist/BeamRider_0000/ckpt_9999.pt ./data/egb_specialist/Berzerk_0000/ckpt_9999.pt ./data/egb_specialist/Bowling_0000/ckpt_9999.pt ./data/egb_specialist/Boxing_0000/ckpt_9999.pt ./data/egb_specialist/Breakout_0000/ckpt_9999.pt ./data/egb_specialist/Centipede_0000/ckpt_9999.pt ./data/egb_specialist/ChopperCommand_0000/ckpt_9999.pt ./data/egb_specialist/CrazyClimber_0000/ckpt_9999.pt ./data/egb_specialist/Defender_0000/ckpt_9999.pt ./data/egb_specialist/DemonAttack_0000/ckpt_9999.pt ./data/egb_specialist/DoubleDunk_0000/ckpt_9999.pt ./data/egb_specialist/FishingDerby_0000/ckpt_9999.pt ./data/egb_specialist/Freeway_0000/ckpt_9999.pt ./data/egb_specialist/Gopher_0000/ckpt_9999.pt ./data/egb_specialist/Gravitar_0000/ckpt_9999.pt ./data/egb_specialist/Hero_0000/ckpt_9999.pt ./data/egb_specialist/IceHockey_0000/ckpt_9999.pt ./data/egb_specialist/Jamesbond_0000/ckpt_9999.pt ./data/egb_specialist/Kangaroo_0000/ckpt_9999.pt ./data/egb_specialist/Krull_0000/ckpt_9999.pt ./data/egb_specialist/KungFuMaster_0000/ckpt_9999.pt ./data/egb_specialist/MsPacman_0000/ckpt_9999.pt ./data/egb_specialist/NameThisGame_0000/ckpt_9999.pt ./data/egb_specialist/Phoenix_0000/ckpt_9999.pt ./data/egb_specialist/Pitfall_0000/ckpt_9999.pt ./data/egb_specialist/Pong_0000/ckpt_9999.pt ./data/egb_specialist/Qbert_0000/ckpt_9999.pt ./data/egb_specialist/Riverraid_0000/ckpt_9999.pt ./data/egb_specialist/RoadRunner_0000/ckpt_9999.pt ./data/egb_specialist/Robotank_0000/ckpt_9999.pt ./data/egb_specialist/Seaquest_0000/ckpt_9999.pt ./data/egb_specialist/Solaris_0000/ckpt_9999.pt ./data/egb_specialist/StarGunner_0000/ckpt_9999.pt ./data/egb_specialist/Surround_0000/ckpt_9999.pt ./data/egb_specialist/Tennis_0000/ckpt_9999.pt ./data/egb_specialist/Tutankham_0000/ckpt_9999.pt ./data/egb_specialist/Venture_0000/ckpt_9999.pt ./data/egb_specialist/VideoPinball_0000/ckpt_9999.pt ./data/egb_specialist/WizardOfWor_0000/ckpt_9999.pt ./data/egb_specialist/YarsRevenge_0000/ckpt_9999.pt ./data/egb_specialist/Zaxxon_0000/ckpt_9999.pt                     --i_split=2 --strategy="final_ckpt"
python bc.py --track=True --project="egb_generalist" --name="all_ckpts_2_0000"  --device="cuda" --model="trans_64" --save_ckpt="./data/egb_generalist/all_ckpts_2_0000/ckpt_{i_iter}.pt"  --env_ids Alien Amidar Assault Asterix Asteroids Atlantis BattleZone BeamRider Berzerk Bowling Boxing Breakout Centipede ChopperCommand CrazyClimber Defender DemonAttack DoubleDunk FishingDerby Freeway Gopher Gravitar Hero IceHockey Jamesbond Kangaroo Krull KungFuMaster MsPacman NameThisGame Phoenix Pitfall Pong Qbert Riverraid RoadRunner Robotank Seaquest Solaris StarGunner Surround Tennis Tutankham Venture VideoPinball WizardOfWor YarsRevenge Zaxxon                     --n_iters=1000 --load_ckpt_teacher ./data/egb_specialist/Alien_0000/ckpt_*.pt ./data/egb_specialist/Amidar_0000/ckpt_*.pt ./data/egb_specialist/Assault_0000/ckpt_*.pt ./data/egb_specialist/Asterix_0000/ckpt_*.pt ./data/egb_specialist/Asteroids_0000/ckpt_*.pt ./data/egb_specialist/Atlantis_0000/ckpt_*.pt ./data/egb_specialist/BattleZone_0000/ckpt_*.pt ./data/egb_specialist/BeamRider_0000/ckpt_*.pt ./data/egb_specialist/Berzerk_0000/ckpt_*.pt ./data/egb_specialist/Bowling_0000/ckpt_*.pt ./data/egb_specialist/Boxing_0000/ckpt_*.pt ./data/egb_specialist/Breakout_0000/ckpt_*.pt ./data/egb_specialist/Centipede_0000/ckpt_*.pt ./data/egb_specialist/ChopperCommand_0000/ckpt_*.pt ./data/egb_specialist/CrazyClimber_0000/ckpt_*.pt ./data/egb_specialist/Defender_0000/ckpt_*.pt ./data/egb_specialist/DemonAttack_0000/ckpt_*.pt ./data/egb_specialist/DoubleDunk_0000/ckpt_*.pt ./data/egb_specialist/FishingDerby_0000/ckpt_*.pt ./data/egb_specialist/Freeway_0000/ckpt_*.pt ./data/egb_specialist/Gopher_0000/ckpt_*.pt ./data/egb_specialist/Gravitar_0000/ckpt_*.pt ./data/egb_specialist/Hero_0000/ckpt_*.pt ./data/egb_specialist/IceHockey_0000/ckpt_*.pt ./data/egb_specialist/Jamesbond_0000/ckpt_*.pt ./data/egb_specialist/Kangaroo_0000/ckpt_*.pt ./data/egb_specialist/Krull_0000/ckpt_*.pt ./data/egb_specialist/KungFuMaster_0000/ckpt_*.pt ./data/egb_specialist/MsPacman_0000/ckpt_*.pt ./data/egb_specialist/NameThisGame_0000/ckpt_*.pt ./data/egb_specialist/Phoenix_0000/ckpt_*.pt ./data/egb_specialist/Pitfall_0000/ckpt_*.pt ./data/egb_specialist/Pong_0000/ckpt_*.pt ./data/egb_specialist/Qbert_0000/ckpt_*.pt ./data/egb_specialist/Riverraid_0000/ckpt_*.pt ./data/egb_specialist/RoadRunner_0000/ckpt_*.pt ./data/egb_specialist/Robotank_0000/ckpt_*.pt ./data/egb_specialist/Seaquest_0000/ckpt_*.pt ./data/egb_specialist/Solaris_0000/ckpt_*.pt ./data/egb_specialist/StarGunner_0000/ckpt_*.pt ./data/egb_specialist/Surround_0000/ckpt_*.pt ./data/egb_specialist/Tennis_0000/ckpt_*.pt ./data/egb_specialist/Tutankham_0000/ckpt_*.pt ./data/egb_specialist/Venture_0000/ckpt_*.pt ./data/egb_specialist/VideoPinball_0000/ckpt_*.pt ./data/egb_specialist/WizardOfWor_0000/ckpt_*.pt ./data/egb_specialist/YarsRevenge_0000/ckpt_*.pt ./data/egb_specialist/Zaxxon_0000/ckpt_*.pt                                                                                                                                                                     --i_split=2 --strategy="all_ckpts" 
