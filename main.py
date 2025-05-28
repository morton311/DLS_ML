# source /home/cmorton/Desktop/beta-Variational-autoencoders-and-transformers-for-reduced-order-modelling-of-fluid-flows/.venv/bin/activate
# python -u main.py -c 're15k' -m 'eval'
import argparse
import os
import shutil
import sys
import torch
import json
import lib.init as init
from lib.runner import runner

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, default='config.json', help='Name or whole path of config file, must be in json format')
parser.add_argument('-o', choices=['l', 'm', 'r', None], default=None, help="Overwrite: 'l' for latent space, 'm' for model, 'r' for results, or None")
parser.add_argument('-m', choices=['train', 'eval', 'pred', 'test'], default='test', help="Mode: 'train' for training, 'eval' for evaluation")
args = parser.parse_args()

# Check if arg contains directory name and/or .json
if '/' not in args.c:
    args.c = 'configs/' + args.c
if '.json' not in args.c:
    args.c = args.c + '.json'

# Check if the config file exists
if not os.path.isfile(args.c):
    raise FileNotFoundError(f"Config file {args.c} not found.")


if __name__ == "__main__":
    # change stdout and stderr to a file named logs/config_name.log
    log_file = args.c.replace('.json', '.log')

    with open(args.c, "r") as f:
        config = json.load(f)
        
    config['overwrite'] = args.o
    config['mode'] = args.m

    device = ('cuda' if torch.cuda.is_available() else "cpu")
    config['device'] = device
    
    run = runner(config)
    if run.config['mode'] == 'train':
        run.train()
    elif run.config['mode'] == 'pred':
        run.pred()
    elif run.config['mode'] == 'eval':
        run.eval()
    

    # copy the config file to the model directory
    shutil.copy(args.c, run.paths_bib.model_dir + os.path.basename(args.c))
    
    
    print(f"{'#'*20}\t{'End of script':<20}\t{'#'*20}")
    # close the log file
    sys.stdout.close()
    sys.stderr.close()
    


