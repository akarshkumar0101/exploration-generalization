

# overfitting to a single batch
python test_e3b_idm.py --n-steps 1e6 --batch-size  256 --lr 5e-4 --freq-batch 1e6 --freq-collect 1e6 --device cuda --track
# overfitting to a single collection
python test_e3b_idm.py --n-steps 1e6 --batch-size  256 --lr 5e-4 --freq-batch   1 --freq-collect 1e6 --device cuda --track
# full training
python test_e3b_idm.py --n-steps 1e6 --batch-size  256 --lr 5e-4 --freq-batch   1 --freq-collect  32 --device cuda --track
python test_e3b_idm.py --n-steps 1e6 --batch-size 1024 --lr 5e-4 --freq-batch   1 --freq-collect  32 --device cuda --track
python test_e3b_idm.py --n-steps 1e6 --batch-size 1024 --lr 1e-4 --freq-batch   1 --freq-collect  32 --device cuda --track
python test_e3b_idm.py --n-steps 1e6 --batch-size 1024 --lr 3e-5 --freq-batch   1 --freq-collect  32 --device cuda --track

