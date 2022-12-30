#!/bin/bash
# 
# Using diffrent receiver spacing
# 
# Run: ./Example-5.sh 

python ./Example-5.py \
--scale_factor=1 \
--until_cmb=y \
--num_epochs=1000 \
--seed=1234 \
--learning_rate=5e-4 \
--rescale_plot=n \
--initial_velocity=3 \
--zid_source=1 \
--zid_receiver=0 \
--data_type=full \
--irregular_grid=y \
--model_type=arid \
--v_multiplier=3 \
--factorization_type=additive \
--tau_act=tanh \
--tau_multiplier=1 \
--max_offset=4.9875 \
--max_depth=1.865625 \
--vertical_spacing=0.009375 \
--lateral_spacing=0.0375 \
--num_neurons=32 \
--num_layers=12 \
--causality_factor=0.5 \
--causality_weight=type_0 \
--reduce_after=50 \
--field_synthetic=n \
--event_factor=0.9 \
--station_factor=0.2 \
--residual_network=n \
--empty_middle=n \
--regular_station=y \
--rec_spacing=20 \
--sou_spacing=20 \
--mixed_precision=y \
--fast_loader=n
