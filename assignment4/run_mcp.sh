#!/bin/bash

python run_experiment.py --seed 42 --policy --verbose --threads 1 > policy.log 2>&1 &
python run_experiment.py --seed 42 --value  --verbose --threads 1 > value.log 2>&1 &
python run_experiment.py --seed 42 --q      --verbose --threads 1 > q.log   2>&1 &

wait 
python run_experiment.py --plot


