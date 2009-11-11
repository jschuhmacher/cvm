#! /usr/bin/env bash

for i in {100.0,10.0,1.0,0.1,0.01,0.001,0.0001,0.00001};
do
    python2.6 -OO ./cerberus.py --header datasets/kddcup.names --trainset datasets/kddcup_500.data --model_out 500_g$i.model --kernel 1 --C 1e6 --gamma $i --log 500_gtr$i.log
    python2.6 -OO ./cerberus.py --header datasets/kddcup.names --testset datasets/kddcup_500.data --model_in 500_g$i.model  --log 500_gte$i.log > scores_g$i.dat
done

for i in {1e-1,1,1e2,1e3,2e2,4e2,5e2,1e5,1e6,1e7};
do
    python2.6 -OO ./cerberus.py --header datasets/kddcup.names --trainset datasets/kddcup_500.data --model_out 500_c$i.model --kernel 1 --C $i --gamma 0.01 --log 500_ctr$i.log
    python2.6 -OO ./cerberus.py --header datasets/kddcup.names --testset datasets/kddcup_500.data --model_in 500_c$i.model --log 500_cte$i.log > scores_c$i.dat
done

for i in {0.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8};
do
    python2.6 -OO ./cerberus.py --header datasets/kddcup.names --trainset datasets/kddcup_500.data --model_out 500_e$i.model --kernel 1 --EPS $i --C 1e4 --gamma 0.01 --log 500_etr$i.log
    python2.6 -OO ./cerberus.py --header datasets/kddcup.names --testset datasets/kddcup_500.data --model_in 500_e$i.model --log 500_ete$i.log > scores_e$i.dat
done
