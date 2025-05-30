#!/bin/bash

python -u main.py -c 're30k_6ktrain' -m 'eval'
echo "Evaluation for re30k_6ktrain completed."
python -u main.py -c 're30k' -m 'eval'
echo "Evaluation for re30k completed."
python -u main.py -c 're15k' -m 'eval'
echo "Evaluation for re15k completed."
python -u main.py -c 're15k' -m 'anim'
echo "Animation for re15k completed."