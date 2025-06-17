# DLS_ML_Polishing

## Research To-do's
- [ ] Lit review writing
- [ ] Formality of sampling efficacy
- [ ] Find qualitative sim studies

## Coding To-do's

- [x] `python -u main.py -c "<config_id>"` 
- [x] Model checkpointing system
- [x] Information passing between scripts
  -  intermediate steps should be saved to disk
- [ ] Performance optimization for batching DLS compression
  - surely there's some optimal batch size?
- [ ] Latent evaluation scripts
  - Comparison of all latent spaces, their energies, reconstruction errors, etc
- [ ] Implement time stamp embedding
- [ ] Support for same latent mode shapes for another data set
- [x] Implement spectograms of point probes
- [ ] Implement PSD contour plots
- [x] Investigate phase portraits for points
- [ ] GPU training logic
  - If array RAM < VRAM available 
    - dset.to(gpu)
  - Else  
    - dset.to(cpu)
- [ ] Implement POD, LSTM comparisons
- [ ] Rapid analysis implementations
- [ ] Change model folder naming scheme to be more readable
  - potential answer in better config naming scheme

<!-- ### Potential Cases on Script Run

- `python -u main.py -o 'latent'`
  - Will need to rerun all results for this latent
- `python -u main.py -m 'train'`
  - model.pth exist
    - `config['overwrite'] = None` -> __crash out__ trained model found, did you mean to overwrite model? 
    - `config['overwrite'] = 'm'` -> train fresh
  - model.pth does not exist
    - checkpoint.pth does not exist -> train fresh
    - checkpoint.pth exists -> load checkpoint, train from checkpoint
- `python -u main.py -m 'eval'`
  - model.pth exists -> load model and eval as normal
  - model.pth does not exists
    - checkpoint.pth does not exist -> train fresh
    - checkpoint.pth exists -> load checkpoint, train from checkpoint, eval as normal -->




## Directory Structure
```
DLS_ML_Polishing/
├── configs/
│   └── <config_id>.json
├── data/
│   └── <data_name>.h5
├── lib/
│   ├── dls.py
│   ├── models.py
│   └── python_func_file_1.py
├── results/
│   └── <data_name>/
│       └── <latent_id>/                # _p{patch_size}_m{num_modes} or POD
│           ├── <model_id>/
│           │   ├── checkpoints/
│           │   ├── figs/
│           │   ├── pred/
│           │   ├── saved_metrics/
│           │   └── <config_id>.json    # copy of config that generated model
│           ├── comp_figs/              # figures comparing models, same latent
│           └── coeffs.h5
├── main.py
├── train.py
├── predict.py
└── eval.py (.ipynb?)
```

## Config .json Structure
```jsonc
<config_id>.json
{
    "data_name":        "<string>",     // e.g., "re30k"

    "latent_type":      "<string>",     // "dls" or "POD"

    "latent_params": {
        "patch_size":   <int>,          // e.g., 19
        "num_modes":    <int>           // e.g., 5
    },

    "model":            "<string>",     // "tr_enc" or "lstm"

    "params": {
        "time_lag":     <int>,          // e.g., 64
        "d_model":      <int>,          // e.g., 512
        "nhead":        <int>,          // e.g., 4
        "num_layers":   <int>           // e.g., 4
    },

    "train": {
        "val_split":    <float>,        // e.g., 0.2 == 20% of all data
        "test_split":   <float>,        // e.g., 0.1 == 10% of train set
        "sample_train": <int>,          // if 0, uses all data in train
        "sample_test":  <int>,          // if 0, uses all data in test
        "lr":           <float>,        // e.g., 0.001
        "num_epochs":   <int>,          // e.g., 1000
        "patience":     <int>,          // e.g., 25
        "train_ahead":  <int>,          // e.g., 5
        "batch_size":   <int>           // e.g., 32
    }
}
```
