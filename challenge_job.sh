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

python3 main.py -c 'challenge/underfit_dls_p25m10' -m 'train'
python3 main.py -c 'challenge/underfit_dls_p25m10' -m 'pred'
python3 main.py -c 'challenge/underfit_dls_p25m10' -m 'eval'
python3 main.py -c 'challenge/underfit_dls_p25m10' -m 'latent'

python3 main.py -c 'challenge/underfit_dls_p25m20' -m 'train'
python3 main.py -c 'challenge/underfit_dls_p25m20' -m 'pred'
python3 main.py -c 'challenge/underfit_dls_p25m20' -m 'eval'
python3 main.py -c 'challenge/underfit_dls_p25m20' -m 'latent'

python3 main.py -c 'challenge/dls_p11m10' -m 'train'
python3 main.py -c 'challenge/dls_p11m10' -m 'pred'
python3 main.py -c 'challenge/dls_p11m10' -m 'eval'
python3 main.py -c 'challenge/dls_p11m10' -m 'latent'