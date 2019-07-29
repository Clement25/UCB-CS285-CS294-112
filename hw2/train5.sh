# !/bin/bash

for b_test in 1000 800 500 200 100 80 50 20 10
do
    for r_test in 5e-3 1e-2 2e-2 5e-2 1e-1 2e-1 s5e-1
    do
        python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $b_test -lr $r_test \
         -rtg --exp_name ip_b{$b_test}_r{$r_test}
    done 
done