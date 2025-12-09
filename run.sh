#!/bin/bash

# python -u main.py -c 're15k/pod/pod_case2_re15k.json' -m 'train'
# python -u main.py -c 're15k/pod/pod_case2_re15k.json' -m 'pred'
# python -u main.py -c 're15k/pod/pod_case2_re15k.json' -m 'eval'
# python -u main.py -c 're15k/pod/pod_case2_re15k.json' -m 'anim'
# python -u main.py -c 're15k/pod/pod_case2_re15k.json' -m 'latent'

# printf "Done with POD re15k case2\n"

# python -u main.py -c 're15k/dls/p49_m10_re15k_ta5.json' -m 'train'
# python -u main.py -c 're15k/dls/p49_m10_re15k_ta5.json' -m 'pred'
# python -u main.py -c 're15k/dls/p49_m10_re15k_ta5.json' -m 'eval'
# python -u main.py -c 're15k/dls/p49_m10_re15k_ta5.json' -m 'anim'
python -u main.py -c 're15k/dls/p49_m10_re15k_ta5.json' -m 'latent'

# printf "Done with DLS re15k p49_m10\n"

# python -u main.py -c 're15k/pod/pod_f_extrap_100mode.json' -m 'train'
# python -u main.py -c 're15k/pod/pod_f_extrap_100mode.json' -m 'pred'
# python -u main.py -c 're15k/pod/pod_f_extrap_100mode.json' -m 'eval'
# python -u main.py -c 're15k/pod/pod_f_extrap_100mode.json' -m 'anim'
# python -u main.py -c 're15k/pod/pod_f_extrap_100mode.json' -m 'latent'

# printf "Done with POD re15k f_extrap_100mode\n"

# python -u main.py -c 're15k/compare.json'
# printf "Done with re15k comparisons\n"

# python -u main.py -c 're30k/pod/pod_case4_re30k.json' -m 'train'
# python -u main.py -c 're30k/pod/pod_case4_re30k.json' -m 'pred'
# python -u main.py -c 're30k/pod/pod_case4_re30k.json' -m 'eval'
# # python -u main.py -c 're30k/pod/pod_case4_re30k.json' -m 'anim'
# python -u main.py -c 're30k/pod/pod_case4_re30k.json' -m 'latent'

# printf "Done with POD re30k case4\n"

# python -u main.py -c 're30k/dls/p49_m10_re30k_ta5.json' -m 'train'
# python -u main.py -c 're30k/dls/p49_m10_re30k_ta5.json' -m 'pred'
# python -u main.py -c 're30k/dls/p49_m10_re30k_ta5.json' -m 'eval'
# # python -u main.py -c 're30k/dls/p49_m10_re15k_ta5.json' -m 'anim'
# python -u main.py -c 're30k/dls/p49_m10_re30k_ta5.json' -m 'latent'

# printf "Done with DLS re30k p49_m10\n"

# python -u main.py -c 're30k/pod/pod_f_extrap_100mode.json' -m 'train'
# # python -u main.py -c 're30k/pod/pod_f_extrap_100mode.json' -m 'pred'
# python -u main.py -c 're30k/pod/pod_f_extrap_100mode.json' -m 'eval'
# # python -u main.py -c 're30k/pod/pod_f_extrap_100mode.json' -m 'anim'
# python -u main.py -c 're30k/pod/pod_f_extrap_100mode.json' -m 'latent'

# printf "Done with POD re30k f_extrap_100mode\n"

# python -u main.py -c 're30k/compare.json'
# printf "Done with re30k comparisons\n"


