#!/bin/bash

python3 -m verl.trainer.main \
    config=examples/config_seed_r1.yaml \
    data.nframes=16 \
    data.resized_height=28 \
    data.resized_width=28

