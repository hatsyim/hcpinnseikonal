#!/bin/bash
# 
# Using diffrent receiver spacing
# 
# Run: ./Example-1.sh 

for i in 5 10 30
do
    python ./Example-1.py \
    --scale_factor=2 \
    --until_cmb=y \
    --num_epochs=3000 \
    --seed=1234 \
    --data_type=nn \
    --learning_rate=1e-3 \
    --rescale_plot=y \
    --initial_velocity=3 \
    --zid_source=5 \
    --zid_receiver=0 \
    --data_type=nn \
    --irregular_grid=y \
    --num_layers=12 \
    --model_type=marmousi \
    --v_multiplier=3 \
    --factorization_type=additive \
    --tau_act=tanh \
    --tau_multiplier=1 \
    --max_offset=8.6 \
    --max_depth=1 \
    --vertical_spacing=0.01 \
    --lateral_spacing=0.03 \
    --num_neurons=24 \
    --causality_factor=.5 \
    --reduce_after=50 \
    --field_synthetic=n \
    --event_factor=0.9 \
    --station_factor=0.1 
    --residual_network=y 
    --empty_middle=n \
    --regular_station=y \
    --sou_spacing=40 \
    --rec_spacing=$i
done
