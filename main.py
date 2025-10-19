# source /home/cmorton/Desktop/beta-Variational-autoencoders-and-transformers-for-reduced-order-modelling-of-fluid-flows/.venv/bin/activate
# sudo nvidia-smi -pl 250
# python -u main.py -c 're30k_p49_m5' -m 'train'
# python -u main.py -c 'pod_case3_re30k' -m 'test'
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
parser.add_argument('-o', choices=['l', 'm', 'r', 'x'], default='x', help="Overwrite: 'l' for latent space, 'm' for model, 'r' for results, or None")
parser.add_argument('-m', choices=['train', 'eval', 'pred', 'test','latent', 'anim'], default='test', help="Mode: 'train' for training, 'eval' for evaluation")
parser.add_argument('-log', choices=['file', 'terminal'], default='file', help="Log output to file or terminal")
args = parser.parse_args()

# Check if arg contains directory name and/or .json
if 'configs/' not in args.c:
    args.c = 'configs/' + args.c
if '.json' not in args.c:
    args.c = args.c + '.json'

# Check if the config file exists
if not os.path.isfile(args.c):
    raise FileNotFoundError(f"Config file {args.c} not found.")


if __name__ == "__main__":

    with open(args.c, "r") as f:
        config = json.load(f)
    
    if "comparisons" in config.keys():
        from lib import comparison 
        from matplotlib import pyplot as plt
        import numpy as np
        num_cases = len(config['comparisons'])

        # make a color map with num_cases colors in plasma
        # colors = plt.cm.cool(np.linspace(0, 1, num_cases))
        # for i in range(num_cases):
        #     config['comparisons'][i]['color'] = colors[i]
            

        for i, key in enumerate(config['comparisons']):
            config['comparisons'][i]['config_file'] = 'configs/' + config['comparisons'][i]['config_file']
            if not os.path.isfile(config['comparisons'][i]['config_file']):
                raise FileNotFoundError(f"Config file {config['comparisons'][i]['config_file']} not found.")
            
            with open(config['comparisons'][i]['config_file'], "r") as f:
                case_config = json.load(f)

            case_config['mode'] = 'compare'
            case_config['overwrite'] = args.o
            
            # remove every field in predictions except the one to be compared
            for j, key in enumerate(list(case_config['predictions'].keys())):
                if key != config['comparisons'][i]['result_key']:
                    case_config['predictions'].pop(key)
            
            device = ('cuda' if torch.cuda.is_available() else "cpu")
            case_config['device'] = device
            case_config['name'] = os.path.basename(config['comparisons'][i]['config_file']).replace('.json', '')


            config['save_dir'] = './results/' + case_config['data_name'] + '/comparison/'
      
            if not os.path.exists(config['save_dir']):
                os.makedirs(config['save_dir'], exist_ok=True)
            if not os.path.exists(config['save_dir'] + 'comparison.log'):
                open(config['save_dir'] + 'comparison.log', 'w').close()
            sys.stdout = open(config['save_dir'] + 'comparison.log', 'a')
            sys.stderr = open(config['save_dir'] + 'comparison.log', 'a')

            run = runner(case_config)
            run.eval()
            config['comparisons'][i]['results'] = run.results

        comparison.compare_RMS(config)
        comparison.compare_tke(config)
        comparison.compare_pdf(config)





    else:
        
        config['overwrite'] = args.o
        config['mode'] = args.m 
        config['name'] = os.path.basename(args.c).replace('.json', '')
        config['log'] = args.log

        device = ('cuda' if torch.cuda.is_available() else "cpu")
        config['device'] = device
        
        run = runner(config)
        if run.config['mode'] == 'train':
            run.train()
        elif run.config['mode'] == 'pred':
            run.pred()
        elif run.config['mode'] == 'eval':
            run.eval()
        elif run.config['mode'] == 'latent':
            from lib.dls import latent_eval
            latent_eval(run)
        elif run.config['mode'] == 'anim':
            from lib.plotting import animate
            animate(run)
        
        # copy the config file to the model directory
        shutil.copy(args.c, run.paths_bib.model_dir + os.path.basename(args.c))
    
    
    print(f"{'#'*20}\t{'End of script':<20}\t{'#'*20}")
    # close the log file
    sys.stdout.close()
    sys.stderr.close()
    


