#!/bin/bash

# python -u main.py -c 're30k_6ktrain' -m 'eval'
# echo "Evaluation for re30k_6ktrain completed."
# python -u main.py -c 're30k' -m 'eval'
# echo "Evaluation for re30k completed."
# python -u main.py -c 're15k' -m 'eval'
# echo "Evaluation for re15k completed."
# python -u main.py -c 're15k' -m 'anim'
# echo "Animation for re15k completed."


# python -u main.py -c 're30k_p49_m5' -m 'pred'
# python -u main.py -c 're30k_p49_m5' -m 'eval'
# python -u main.py -c 're30k_p49_m5' -m 'anim'

# python -u main.py -c 'half_p49_m10_re30k' -m 'latent'
# python -u main.py -c 'half_p49_m10_re30k' -m 'train'
# python -u main.py -c 'half_p49_m10_re30k' -m 'pred'
# python -u main.py -c 'half_p49_m10_re30k' -m 'eval'
# python -u main.py -c 'half_p49_m10_re30k' -m 'anim'

# python -u main.py -c 'bvae_case1_re15k' -m 'latent'
# python -u main.py -c 'bvae_case1_re15k' -m 'train'
python -u main.py -c 'bvae_case1_re15k' -m 'pred'
python -u main.py -c 'bvae_case1_re15k' -m 'eval'
# python -u main.py -c 'bvae_case1_re15k' -m 'anim' 