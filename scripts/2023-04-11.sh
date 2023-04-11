# full training
python test_e3b_idm.py --n-steps 1e5 --seed 0 --batch-size 256 --lr 5e-4 --idm-merge cat --freq-batch   1 --freq-collect  32 --device cuda --track
python test_e3b_idm.py --n-steps 1e5 --seed 0 --batch-size 256 --lr 5e-4 --idm-merge sub --freq-batch   1 --freq-collect  32 --device cuda --track
python test_e3b_idm.py --n-steps 1e5 --seed 1 --batch-size 256 --lr 5e-4 --idm-merge cat --freq-batch   1 --freq-collect  32 --device cuda --track
python test_e3b_idm.py --n-steps 1e5 --seed 1 --batch-size 256 --lr 5e-4 --idm-merge sub --freq-batch   1 --freq-collect  32 --device cuda --track
python test_e3b_idm.py --n-steps 1e5 --seed 2 --batch-size 256 --lr 5e-4 --idm-merge cat --freq-batch   1 --freq-collect  32 --device cuda --track
python test_e3b_idm.py --n-steps 1e5 --seed 2 --batch-size 256 --lr 5e-4 --idm-merge sub --freq-batch   1 --freq-collect  32 --device cuda --track