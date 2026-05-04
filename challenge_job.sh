#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=challenge_job
#SBATCH --account=NAWCP24632466
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 02:00:00
#SBATCH --output=out_challenge.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

python3 main.py -c 'challenge/dls/p25m10/swiglu' -m 'train'
python3 main.py -c 'challenge/dls/p25m10/swiglu' -m 'pred'
python3 main.py -c 'challenge/dls/p25m10/swiglu' -m 'eval'
python3 main.py -c 'challenge/dls/p25m10/swiglu' -m 'latent'

# python3 main.py -c 'challenge/dls/p25m10/small' -m 'train'
# python3 main.py -c 'challenge/dls/p25m10/small' -m 'pred'
# python3 main.py -c 'challenge/dls/p25m10/small' -m 'eval'
# python3 main.py -c 'challenge/dls/p25m10/small' -m 'latent'

# python3 main.py -c 'challenge/dls/p25m20/small' -m 'train'
# python3 main.py -c 'challenge/dls/p25m20/small' -m 'pred'
# python3 main.py -c 'challenge/dls/p25m20/small' -m 'eval'
# python3 main.py -c 'challenge/dls/p25m20/small' -m 'latent'

# python3 main.py -c 'challenge/dls/p25m10/underfit' -m 'train'
# python3 main.py -c 'challenge/dls/p25m10/underfit' -m 'pred'
# python3 main.py -c 'challenge/dls/p25m10/underfit' -m 'eval'
# python3 main.py -c 'challenge/dls/p25m10/underfit' -m 'latent'

# python3 main.py -c 'challenge/dls/p25m10/underfit' -m 'train'
# python3 main.py -c 'challenge/dls/p25m20/underfit' -m 'pred'
# python3 main.py -c 'challenge/dls/p25m20/underfit' -m 'eval'
# python3 main.py -c 'challenge/dls/p25m20/underfit' -m 'latent'

# python3 main.py -c 'challenge/dls/p11m10/case' -m 'train'
# python3 main.py -c 'challenge/dls/p11m10/case' -m 'pred'
# python3 main.py -c 'challenge/dls/p11m10/case' -m 'eval'
# python3 main.py -c 'challenge/dls/p11m10/case' -m 'latent'