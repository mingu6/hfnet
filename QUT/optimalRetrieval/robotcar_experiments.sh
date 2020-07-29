#!/bin/bash/

alld=(5 10 20 50 100)
allk=(1 2 3 5)

for k in ${allk[@]}; do
    for d in ${alld[@]}; do
        python3 evaluate_robotcar.py hfnet_model robotcar --local_method hfnet --global_method hfnet --queries night --export_poses --num_nearest $k --num_distractors $d --imperfect
        python3 evaluate_robotcar.py hfnet_model robotcar --local_method hfnet --global_method hfnet --queries night-rain --export_poses --num_nearest $k --num_distractors $d --imperfect
    done
done
