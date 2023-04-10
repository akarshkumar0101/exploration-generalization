

# overfitting to a single batch
python test_e3b_idm.py --n-steps 1e6 --batch-size  256 --freq-batch 1e6 --freq-collect 1e6 --device cuda:0 --track
# overfitting to a single collection
python test_e3b_idm.py --n-steps 1e6 --batch-size  256 --freq-batch   1 --freq-collect 1e6 --device cuda:1 --track
# full training
python test_e3b_idm.py --n-steps 1e6 --batch-size  256 --freq-batch   1 --freq-collect  32 --device cuda:2 --track
python test_e3b_idm.py --n-steps 1e6 --batch-size 1024 --freq-batch   1 --freq-collect  32 --device cuda:3 --track
python test_e3b_idm.py --n-steps 1e6 --batch-size 2048 --freq-batch   1 --freq-collect  32 --device cuda:4 --track


