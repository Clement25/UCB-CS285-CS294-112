# !/bin/bash

python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 \
-l 2 -s 64 -b <b*> -lr <r*> -rtg --exp_name ip_b<b*>_r<r*>