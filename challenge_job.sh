#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=challenge_job
#SBATCH --account=NAWCP24632466
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 02:00:00
#SBATCH --output=out_challenge.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

#python3 main.py -c 'challenge/dls_tr_ta1_2_1' -m 'train'
#python3 main.py -c 'challenge/dls_tr_ta1_2_1' -m 'pred'
python3 main.py -c 'challenge/dls_tr_ta1_2_1' -m 'eval'
python3 main.py -c 'challenge/dls_tr_ta1_2_1' -m 'latent'
