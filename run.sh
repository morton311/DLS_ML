#!/bin/bash


# python -u main.py -c 'plate/bvae/bvae_plate_same_params.json' -m 'latent'
# python -u main.py -c 're30k/pod/pod_f_extrap_100mode.json' -m 'latent'
# python -u main.py -c 're30k/pod/pod_f_extrap_100mode.json' -m 'train'
# python -u main.py -c 're30k/pod/pod_f_extrap_100mode.json' -m 'pred'
# python -u main.py -c 're30k/pod/pod_f_extrap_100mode.json' -m 'eval'

python -u main.py -c 're15k/compare.json'

# python -u main.py -c 're15k/pod/pod_f_extrap_100mode.json' -m 'eval'
# python -u main.py -c 're15k/pod/pod_f_extrap_100mode.json' -m 'anim'

