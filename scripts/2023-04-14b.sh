

python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge cat      --idm-normalize True  --actions ordinal --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2
python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge cat      --idm-normalize False --actions ordinal --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2

python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge diff     --idm-normalize True  --actions ordinal --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2
python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge diff     --idm-normalize False --actions ordinal --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2

python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge both     --idm-normalize True  --actions ordinal --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2
python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge both     --idm-normalize False --actions ordinal --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2

python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge catdinit --idm-normalize True  --actions ordinal --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2
python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge catdinit --idm-normalize False --actions ordinal --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2


python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge cat      --idm-normalize True  --actions all     --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2
python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge cat      --idm-normalize False --actions all     --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2

python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge diff     --idm-normalize True  --actions all     --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2
python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge diff     --idm-normalize False --actions all     --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2

python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge both     --idm-normalize True  --actions all     --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2
python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge both     --idm-normalize False --actions all     --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2

python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge catdinit --idm-normalize True  --actions all     --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2
python test_e3b_idm.py --lr 5e-4 --n-steps 5e4 --batch-size 2048 --idm-merge catdinit --idm-normalize False --actions all     --seed 0 --device cuda --track --name e3bidmtest_{idm_merge}_{idm_normalize}_{actions}_{seed} --project e3b_idm_test2

