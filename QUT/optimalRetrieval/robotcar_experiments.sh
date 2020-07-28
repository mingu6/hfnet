#!/bin/bash/

allk=(1 3 5 10)
alld=(0 5 10 20)

for k in ${allk[@]}; do
    for d in ${alld[@]}; do
        python3 evaluate_robotcar.py hfnet_model robotcar --local_method hfnet --global_method hfnet --queries night --export_poses --num_nearest $k --num_distractors $d
        python3 evaluate_robotcar.py hfnet_model robotcar --local_method hfnet --global_method hfnet --queries night-rain --export_poses --num_nearest $k --num_distractors $d
    done
done
