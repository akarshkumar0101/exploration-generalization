python pretrain.py --pretrain-levels 00001 --pretrain-obj ext --device cuda:0 --track &
python pretrain.py --pretrain-levels 00001 --pretrain-obj int --device cuda:1 --track &
python pretrain.py --pretrain-levels 00002 --pretrain-obj ext --device cuda:2 --track &
python pretrain.py --pretrain-levels 00002 --pretrain-obj int --device cuda:3 --track &
wait
python pretrain.py --pretrain-levels 00004 --pretrain-obj ext --device cuda:0 --track &
python pretrain.py --pretrain-levels 00004 --pretrain-obj int --device cuda:1 --track &
python pretrain.py --pretrain-levels 00008 --pretrain-obj ext --device cuda:2 --track &
python pretrain.py --pretrain-levels 00008 --pretrain-obj int --device cuda:3 --track &
wait
python pretrain.py --pretrain-levels 00016 --pretrain-obj ext --device cuda:0 --track &
python pretrain.py --pretrain-levels 00016 --pretrain-obj int --device cuda:1 --track &
python pretrain.py --pretrain-levels 00032 --pretrain-obj ext --device cuda:2 --track &
python pretrain.py --pretrain-levels 00032 --pretrain-obj int --device cuda:3 --track &
wait


python train.py --level 10000 --train-obj ext --pretrain-levels 00001 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10000 --train-obj ext --pretrain-levels 00001 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10000 --train-obj ext --pretrain-levels 00002 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10000 --train-obj ext --pretrain-levels 00002 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10000 --train-obj ext --pretrain-levels 00004 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10000 --train-obj ext --pretrain-levels 00004 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10000 --train-obj ext --pretrain-levels 00008 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10000 --train-obj ext --pretrain-levels 00008 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10000 --train-obj ext --pretrain-levels 00016 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10000 --train-obj ext --pretrain-levels 00016 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10000 --train-obj ext --pretrain-levels 00032 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10000 --train-obj ext --pretrain-levels 00032 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10001 --train-obj ext --pretrain-levels 00001 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10001 --train-obj ext --pretrain-levels 00001 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10001 --train-obj ext --pretrain-levels 00002 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10001 --train-obj ext --pretrain-levels 00002 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10001 --train-obj ext --pretrain-levels 00004 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10001 --train-obj ext --pretrain-levels 00004 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10001 --train-obj ext --pretrain-levels 00008 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10001 --train-obj ext --pretrain-levels 00008 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10001 --train-obj ext --pretrain-levels 00016 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10001 --train-obj ext --pretrain-levels 00016 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10001 --train-obj ext --pretrain-levels 00032 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10001 --train-obj ext --pretrain-levels 00032 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10002 --train-obj ext --pretrain-levels 00001 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10002 --train-obj ext --pretrain-levels 00001 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10002 --train-obj ext --pretrain-levels 00002 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10002 --train-obj ext --pretrain-levels 00002 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10002 --train-obj ext --pretrain-levels 00004 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10002 --train-obj ext --pretrain-levels 00004 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10002 --train-obj ext --pretrain-levels 00008 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10002 --train-obj ext --pretrain-levels 00008 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10002 --train-obj ext --pretrain-levels 00016 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10002 --train-obj ext --pretrain-levels 00016 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10002 --train-obj ext --pretrain-levels 00032 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10002 --train-obj ext --pretrain-levels 00032 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10003 --train-obj ext --pretrain-levels 00001 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10003 --train-obj ext --pretrain-levels 00001 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10003 --train-obj ext --pretrain-levels 00002 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10003 --train-obj ext --pretrain-levels 00002 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10003 --train-obj ext --pretrain-levels 00004 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10003 --train-obj ext --pretrain-levels 00004 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10003 --train-obj ext --pretrain-levels 00008 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10003 --train-obj ext --pretrain-levels 00008 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10003 --train-obj ext --pretrain-levels 00016 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10003 --train-obj ext --pretrain-levels 00016 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10003 --train-obj ext --pretrain-levels 00032 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10003 --train-obj ext --pretrain-levels 00032 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10004 --train-obj ext --pretrain-levels 00001 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10004 --train-obj ext --pretrain-levels 00001 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10004 --train-obj ext --pretrain-levels 00002 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10004 --train-obj ext --pretrain-levels 00002 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10004 --train-obj ext --pretrain-levels 00004 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10004 --train-obj ext --pretrain-levels 00004 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10004 --train-obj ext --pretrain-levels 00008 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10004 --train-obj ext --pretrain-levels 00008 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10004 --train-obj ext --pretrain-levels 00016 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10004 --train-obj ext --pretrain-levels 00016 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10004 --train-obj ext --pretrain-levels 00032 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10004 --train-obj ext --pretrain-levels 00032 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10005 --train-obj ext --pretrain-levels 00001 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10005 --train-obj ext --pretrain-levels 00001 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10005 --train-obj ext --pretrain-levels 00002 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10005 --train-obj ext --pretrain-levels 00002 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10005 --train-obj ext --pretrain-levels 00004 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10005 --train-obj ext --pretrain-levels 00004 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10005 --train-obj ext --pretrain-levels 00008 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10005 --train-obj ext --pretrain-levels 00008 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10005 --train-obj ext --pretrain-levels 00016 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10005 --train-obj ext --pretrain-levels 00016 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10005 --train-obj ext --pretrain-levels 00032 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10005 --train-obj ext --pretrain-levels 00032 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10006 --train-obj ext --pretrain-levels 00001 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10006 --train-obj ext --pretrain-levels 00001 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10006 --train-obj ext --pretrain-levels 00002 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10006 --train-obj ext --pretrain-levels 00002 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10006 --train-obj ext --pretrain-levels 00004 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10006 --train-obj ext --pretrain-levels 00004 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10006 --train-obj ext --pretrain-levels 00008 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10006 --train-obj ext --pretrain-levels 00008 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10006 --train-obj ext --pretrain-levels 00016 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10006 --train-obj ext --pretrain-levels 00016 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10006 --train-obj ext --pretrain-levels 00032 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10006 --train-obj ext --pretrain-levels 00032 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10007 --train-obj ext --pretrain-levels 00001 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10007 --train-obj ext --pretrain-levels 00001 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10007 --train-obj ext --pretrain-levels 00002 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10007 --train-obj ext --pretrain-levels 00002 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10007 --train-obj ext --pretrain-levels 00004 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10007 --train-obj ext --pretrain-levels 00004 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10007 --train-obj ext --pretrain-levels 00008 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10007 --train-obj ext --pretrain-levels 00008 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10007 --train-obj ext --pretrain-levels 00016 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10007 --train-obj ext --pretrain-levels 00016 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10007 --train-obj ext --pretrain-levels 00032 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10007 --train-obj ext --pretrain-levels 00032 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10008 --train-obj ext --pretrain-levels 00001 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10008 --train-obj ext --pretrain-levels 00001 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10008 --train-obj ext --pretrain-levels 00002 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10008 --train-obj ext --pretrain-levels 00002 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10008 --train-obj ext --pretrain-levels 00004 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10008 --train-obj ext --pretrain-levels 00004 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10008 --train-obj ext --pretrain-levels 00008 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10008 --train-obj ext --pretrain-levels 00008 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10008 --train-obj ext --pretrain-levels 00016 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10008 --train-obj ext --pretrain-levels 00016 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10008 --train-obj ext --pretrain-levels 00032 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10008 --train-obj ext --pretrain-levels 00032 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10009 --train-obj ext --pretrain-levels 00001 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10009 --train-obj ext --pretrain-levels 00001 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10009 --train-obj ext --pretrain-levels 00002 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10009 --train-obj ext --pretrain-levels 00002 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10009 --train-obj ext --pretrain-levels 00004 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10009 --train-obj ext --pretrain-levels 00004 --pretrain-obj int --device cuda:2 --track &
wait
python train.py --level 10009 --train-obj ext --pretrain-levels 00008 --pretrain-obj ext --device cuda:0 --track &
python train.py --level 10009 --train-obj ext --pretrain-levels 00008 --pretrain-obj int --device cuda:0 --track &
python train.py --level 10009 --train-obj ext --pretrain-levels 00016 --pretrain-obj ext --device cuda:1 --track &
python train.py --level 10009 --train-obj ext --pretrain-levels 00016 --pretrain-obj int --device cuda:1 --track &
python train.py --level 10009 --train-obj ext --pretrain-levels 00032 --pretrain-obj ext --device cuda:2 --track &
python train.py --level 10009 --train-obj ext --pretrain-levels 00032 --pretrain-obj int --device cuda:2 --track &
wait

