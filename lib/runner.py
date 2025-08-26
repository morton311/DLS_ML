import os
import sys
from directory_tree import DisplayTree
import h5py
import pickle
import copy

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import time
import builtins
from functools import partial
print = partial(print, flush=True)
builtins.print = print

import lib.init as init
import lib.dls as dls
import lib.pod as pod
import lib.models as models
import lib.datas as datas


class runner(nn.Module):
    def __init__(self, config):
        super(runner, self).__init__()
        self.config = config
        self.device = config['device']
        self.paths_bib = self._init_paths_and_logging(config)

        self._log_config()

        # get model info
        self._get_data()
        self._get_model()
        self._compile_model()
    

    def _init_paths_and_logging(self, config):
        is_init_path, paths = init.init_path(config)

        if config['log'] == 'file':
            sys.stdout = open(paths.log_path, 'w')
            sys.stderr = open(paths.log_path, 'a')
        

        print(f'Using device: {self.device}')
        if is_init_path:
            DisplayTree(header=True, ignoreList=['*.pyc', '*.png'], maxDepth=4)
        return paths


    def _log_config(self):
        print(f"{'#'*20} Configuration {'#'*20}")
        for key, val in self.config.items():
            if isinstance(val, dict):
                print(f"{key}:")
                for sub_key, sub_val in val.items():
                    print(f"  {sub_key}: {sub_val}")
            else:
                print(f"{key}: {val}")


    def _get_data(self):
        """
        Load the latent coefficients
        """
        print(f"{'#'*20}\t{'Loading data...':<20}\t{'#'*20}")
        
        if not os.path.exists(self.paths_bib.latent_path) or self.config['overwrite'] == 'l':
            self._compute_latent_coefficients()
        self._latent_split()

        # load latent_config 
        with open(self.paths_bib.latent_path.replace('.h5', '_config.pkl'), 'rb') as f:
            self.l_config = pickle.load(f)


    def _compute_latent_coefficients(self):
        print("Computing latent coefficients...")
        with h5py.File(self.paths_bib.data_path, 'r') as f:
            num_snaps = f['UV'].shape[0]

        if self.config['latent_type'] == 'dls':
            latent_config = dls.gfem_2d_long(
                data_path=self.paths_bib.data_path,
                field_name='UV',
                latent_file=self.paths_bib.latent_path,
                patch_size=self.config['latent_params']['patch_size'],
                num_modes=self.config['latent_params']['num_modes'],
                batch_size=num_snaps // 20
            )

            with open(self.paths_bib.latent_path.replace('.h5', '_config.pkl'), 'wb') as f:
                pickle.dump(latent_config, f)
            print("Latent coefficient config saved")

        elif self.config['latent_type'] == 'pod':
            with h5py.File(self.paths_bib.data_path, 'r') as f:
                mean = f['mean'][:]
                data = f['UV'][:2500:2] - mean[np.newaxis, ...]
            modes, eigVal, latent_config = pod.pod_mode_find(data)

            with h5py.File(self.paths_bib.latent_path, 'w') as f:
                f.create_dataset('eigVal', data=eigVal)
                f.create_dataset('modes', data=modes)

            pod.pod_decomp(self.paths_bib.data_path, self.paths_bib.latent_path)
                
            with open(self.paths_bib.latent_path.replace('.h5', '_config.pkl'), 'wb') as f:
                pickle.dump(latent_config, f)

        elif self.config['latent_type'] == 'bvae':
            bvae = models.bvae_model(latent_dim=self.config['latent_params']['latent_dim'])
            bvae.to(self.device)

            if not os.path.exists(self.paths_bib.latent_model_path) and not self.config['overwrite'] in ['l', 'm']:

                train_snaps = self.config['latent_params'].get('train_snaps', 2500)
                train_split = self.config['latent_params'].get('train_test_split', 0.8)
                test_split = self.config['latent_params'].get('train_val_split', 0.1)

                
                with h5py.File(self.paths_bib.data_path, 'r') as f:
                    total_snaps = f['UV'].shape[0]
                    if total_snaps < train_snaps:
                        train_snaps = total_snaps

                    train_len = int(train_snaps * train_split)
                    test_len = int(train_len * test_split)

                    train_indices = np.arange(0, train_len)
                    test_indices = np.arange(train_len, train_len + test_len)
                    val_indices = np.arange(train_len + test_len, total_snaps)

                    mean = f['mean'][:]
                    train_set = np.array(f['UV'][train_indices] - mean[np.newaxis, ...])
                    test_set = np.array(f['UV'][test_indices] - mean[np.newaxis, ...])

                    # compute mean and std of train set and save to latent_dir/latent_scaler.pkl
                    train_mean = np.mean(train_set, axis=0)
                    train_std = np.std(train_set, axis=0)
                    with open(self.paths_bib.latent_dir + 'latent_scaler.pkl', 'wb') as f:
                        pickle.dump((train_mean, train_std), f)

                    train_set = datas.normalize_data(train_set, train_mean, train_std)
                    test_set = datas.normalize_data(test_set, train_mean, train_std)

                print(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")

                train_set = train_set.transpose(0, 3, 1, 2)  # [S, C, H, W]
                test_set = test_set.transpose(0, 3, 1, 2)    # [S, C, H, W]

                # make train and test data loaders
                train_loader = datas.make_dataloader(
                    torch.from_numpy(train_set).float().to(self.device),
                    torch.from_numpy(train_set).float().to(self.device),
                    batch_size=self.config['latent_params'].get('batch_size', 256),
                    shuffle=True
                )
                test_loader = datas.make_dataloader(
                    torch.from_numpy(test_set).float().to(self.device),
                    torch.from_numpy(test_set).float().to(self.device),
                    batch_size=self.config['latent_params'].get('batch_size', 256),
                    shuffle=False
                )
                

                # create optimizer
                optimizer = torch.optim.Adam(bvae.parameters(), lr=self.config['latent_params'].get('lr', 2e-4))

                bvae, losses = models.train_bvae(
                    model=bvae,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    optimizer=optimizer,
                    config=self.config
                )

                # Save model and losses
                torch.save(bvae.state_dict(), self.paths_bib.latent_model_path)
                with open(self.paths_bib.latent_dir + 'bvae_losses.pkl', 'wb') as f:
                    pickle.dump(losses, f)
                
                print(f"Latent model saved to {self.paths_bib.latent_model_path}")
                print(f"Latent losses saved to {self.paths_bib.latent_dir + 'bvae_losses.pkl'}")

            else:
                print(f"Latent model already exists at {self.paths_bib.latent_model_path}. Loading model.")
                
                bvae.load_state_dict(torch.load(self.paths_bib.latent_model_path, weights_only=True))
                bvae.to(self.device)


            if os.path.exists(self.paths_bib.latent_path):
                print(f"Latent coefficients already exist at {self.paths_bib.latent_path}. Skipping encoding.")
            else:
                # Encode the full dataset to get latent coefficients
                models.bvae_batch_encode(
                    model=bvae,
                    data_path=self.paths_bib.data_path,
                    latent_path=self.paths_bib.latent_path,
                    config=self.config,
                    device=self.device
                )


    def _latent_split(self):
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            if self.config['latent_type'] == 'dls':
                input_dim = 2 * f['dof_u'].shape[1]
                total_snaps = f['dof_u'].shape[0]
            elif self.config['latent_type'] == 'pod':
                total_snaps = f['dofs'].shape[0]
                input_dim = self.config['latent_params']['num_modes'] 
            elif self.config['latent_type'] == 'bvae':
                total_snaps = f['dofs'].shape[0]
                input_dim = self.config['latent_params']['latent_dim']
            self.config['params']['input_dim'] = input_dim
            

        self._split_indices(total_snaps)


    def _split_indices(self, total_snaps):
        # find indices for train, test, and validation sets
        train_len = int(total_snaps * self.config['train']['train_split'])
        test_len = int(train_len * self.config['train']['test_split'])
        
        train_indices = np.arange(0, train_len)
        test_indices = np.arange(train_len, train_len + test_len)
        val_indices = np.arange(train_len + test_len, total_snaps)
        
        # Print a table with the set info
        sample_train = self.config['train']['sample_train']
        sample_test = self.config['train']['sample_test']

        print(f"{'Set':<12}|{'Total':<10}|{'First Idx':<12}|{'Last Idx':<12}|{'Sampled':<10}")
        print("-" * 56)
        print(f"{'Train':<12}|{train_len:<10}|{train_indices[0]:<12}|{train_indices[-1]:<12}|{sample_train:<10}")
        print(f"{'Test':<12}|{len(test_indices):<10}|{test_indices[0]:<12}|{test_indices[-1]:<12}|{sample_test:<10}")
        print(f"{'Validation':<12}|{len(val_indices):<10}|{val_indices[0]:<12}|{val_indices[-1]:<12}|{'-':<10}")

        if not os.path.exists(self.paths_bib.model_dir + 'split_ids.pkl'):


            self.train_indices = np.sort(datas.sample_series_indices(
                                        train_len, 
                                        sample_train, 
                                        time_lag=self.config['params']['time_lag'], 
                                        train_ahead=self.config['train']['train_ahead'], 
                                        seed=42))
            self.test_indices = np.sort(datas.sample_series_indices(
                                        len(test_indices), 
                                        sample_test, 
                                        time_lag=self.config['params']['time_lag'], 
                                        train_ahead=self.config['train']['train_ahead'], 
                                        seed=42))
            self.val_indices = val_indices

            # save the train, test, and validation indices
            with open(self.paths_bib.model_dir + 'split_ids.pkl', 'wb') as f:
                pickle.dump({'train_indices': self.train_indices, 'test_indices': self.test_indices, 'val_indices': self.val_indices}, f)
            print(f"Train, test, and validation indices saved to {self.paths_bib.model_dir + 'split_ids.pkl'}")

        else:
            with open(self.paths_bib.model_dir + 'split_ids.pkl', 'rb') as f:
                indices = pickle.load(f)
                self.train_indices = indices['train_indices']
                self.test_indices = indices['test_indices']
                self.val_indices = indices['val_indices']
            print(f"Train, test, and validation indices loaded from {self.paths_bib.model_dir + 'split_ids.pkl'}")


    def _get_model(self):
        """
        Load the model
        """
        # Load the model
        print(f"{'#'*20}\t{'Loading model...':<20}\t{'#'*20}")
        if self.config['model'] == 'tr_enc':
            self.model = models.TransformerEncoderModel(
                        time_lag=self.config['params']['time_lag'],
                        input_dim=self.config['params']['input_dim'],
                        d_model=self.config['params']['d_model'],
                        nhead=self.config['params']['nhead'],
                        num_layers=self.config['params']['num_layers'],
                        embed=self.config['params'].get('embed', 'lin')
                        ).to(self.device)
        elif self.config['model'] == 'lstm':
            self.model = models.LSTMModel(
                        time_lag=self.config['params']['time_lag'],
                        input_dim=self.config['params']['input_dim'],
                        hidden_dim=self.config['params']['d_model'],
                        num_layers=self.config['params']['num_layers'],
                        batch_size= self.config['train']['batch_size'],
                        ).to(self.device)
        else:
            raise ValueError(f"Model {self.config['model']} not recognized. Please use 'tr_enc' or 'lstm'.")
        
        # Load the model weights if they exist and overwrite is not set to 'l' or 'm'
        if os.path.exists(self.paths_bib.model_path) and not self.config['overwrite'] in ['l', 'm']:
            self.model.load_state_dict(torch.load(self.paths_bib.model_path, weights_only=True))
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")


    def _compile_model(self):
        """
        Compile the model
        """
        # Define the loss function and optimizer
        print(f"{'#'*20}\t{'Compiling model...':<20}\t{'#'*20}")
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['train']['lr'])

        print(f"Loss function: {self.criterion}")
        print(f"Optimizer: {self.optimizer}")

        # Helper function to remap 'embed' keys to 'input_projection'
        def remap_embed_keys(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('embed'):
                    new_k = k.replace('embed', 'input_projection', 1)
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            return new_state_dict

        # if checkpoint file exists, model doesn't exist, and overwrite is not set to 'l' or 'm', load the checkpoint
        check_flag = os.path.exists(self.paths_bib.checkpoint_path)
        model_flag = os.path.exists(self.paths_bib.model_path)
        if check_flag and not model_flag and not self.config['overwrite'] in ['l', 'm']: 
            print(f"Loading checkpoint from {self.paths_bib.checkpoint_path}")
            checkpoint = torch.load(self.paths_bib.checkpoint_path, weights_only=True)
            checkpoint['model_state_dict'] = remap_embed_keys(checkpoint['model_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.losses = checkpoint['losses']
            self.test_losses = checkpoint['test_losses']
            self.early_stop_counter = checkpoint['early_stop_counter']
            print(f"Checkpoint loaded")
            self.checkpointed = True

        # if model exists and overwrite is not set to 'l' or 'm', load the model and skip training
        elif model_flag and not self.config['overwrite'] in ['l', 'm']:
            print(f"Model already exists at {self.paths_bib.model_path}. Skipping training.")
            print(f"Loading model weights from {self.paths_bib.model_path}")
            state_dict = torch.load(self.paths_bib.model_path, weights_only=True)
            state_dict = remap_embed_keys(state_dict)
            self.model.load_state_dict(state_dict)
            self.checkpointed = False
        else:
            print(f"Model does not exist at {self.paths_bib.model_path}. Training from scratch.")
            self.checkpointed = False


    def train(self):
        """
        Train the model
        """
        # Load the latent coefficients
        print(f"{'#'*20}\t{'Training model...':<20}\t{'#'*20}")
        self._get_train_data()
        self._model_fit()
        

    def _get_train_data(self):
        """
        Get training and test data as torch tensors, minimizing memory usage.
        """
        print('Getting training and test data')
        tl = self.config['params']['time_lag']
        ta = self.config['train']['train_ahead']
        dof_dim = self.config['params']['input_dim']
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            if self.config['latent_type'] == 'dls':
                dof_u = f['dof_u']
                dof_v = f['dof_v']
                # Compute mean and std using only the training indices, in chunks to save memory
                dofs = torch.zeros(len(self.train_indices), dof_dim)
                for i, idx in enumerate(self.train_indices):
                    u = torch.from_numpy(dof_u[idx:idx+1]).float()
                    v = torch.from_numpy(dof_v[idx:idx+1]).float()
                    dofs[i] = torch.cat((u, v), dim=1)
            elif self.config['latent_type'] == 'pod':
                dofs_not_scaled = f['dofs']
                dofs = torch.zeros(len(self.train_indices), dof_dim)
                for i, idx in enumerate(self.train_indices):
                    dofs[i] = torch.from_numpy(dofs_not_scaled[idx:idx+1, :dof_dim]).float()

            elif self.config['latent_type'] == 'bvae':
                dofs_not_scaled = f['dofs']
                dofs = torch.zeros(len(self.train_indices), dof_dim)
                for i, idx in enumerate(self.train_indices):
                    dofs[i] = torch.from_numpy(dofs_not_scaled[idx:idx+1, :dof_dim]).float()

            dof_mean = torch.mean(dofs, dim=0)
            dof_std = torch.std(dofs, dim=0)

            # print(f"Mean of dof: {dof_mean}, Std of dof: {dof_std}")

            self.dof_mean = dof_mean
            self.dof_std = dof_std

            with open(os.path.join(self.paths_bib.model_dir, 'dof_scaler.pkl'), 'wb') as f:
                pickle.dump((dof_mean, dof_std), f)

            # Helper to get normalized dof sequence as torch tensor
            def get_dof_seq(idx, length, latent_type='dls'):
                if latent_type == 'dls':
                    u = torch.from_numpy(dof_u[idx:idx+length]).float()
                    v = torch.from_numpy(dof_v[idx:idx+length]).float()
                    dof = torch.cat((u, v), dim=1)
                elif latent_type == 'pod':
                    dof = torch.from_numpy(dofs_not_scaled[idx:idx+length, :dof_dim]).float()
                elif latent_type == 'bvae':
                    dof = torch.from_numpy(dofs_not_scaled[idx:idx+length, :dof_dim]).float()
                dof = (dof - dof_mean) / dof_std
                return dof

            # Prepare lists for X/Y, then stack at the end
            X_train, Y_train = torch.zeros(len(self.train_indices), tl, dof_dim), torch.zeros(len(self.train_indices), ta, dof_dim)
            for i, idx in enumerate(self.train_indices):
                dof_seq = get_dof_seq(idx, tl + ta, latent_type=self.config['latent_type'])
                X_train[i] = dof_seq[:tl]
                Y_train[i] = dof_seq[tl:tl+ta]
                if i % 500 == 0:
                    print(f"Got train data {i}/{len(self.train_indices)}")
            print('Got train data')

            X_test, Y_test = torch.zeros(len(self.test_indices), tl, dof_dim), torch.zeros(len(self.test_indices), ta, dof_dim)
            for i, idx in enumerate(self.test_indices):
                dof_seq = get_dof_seq(idx, tl + ta, latent_type=self.config['latent_type'])
                X_test[i] = dof_seq[:tl]
                Y_test[i] = dof_seq[tl:tl+ta]
                if i % 100 == 0:
                    print(f"Got test data {i}/{len(self.test_indices)}")
            print('Got test data')


        print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}, dtype: {X_train.dtype}")
        print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}, dtype: {X_test.dtype}")

        # convert to data loader
        self.train_loader = datas.make_dataloader(X_train.to(self.device), Y_train.to(self.device), batch_size=self.config['train']['batch_size'], shuffle=True)
        print(f"Train loader created with {len(self.train_loader)} batches")
        self.test_loader = datas.make_dataloader(X_test.to(self.device), Y_test.to(self.device), batch_size=self.config['train']['batch_size'], shuffle=False)
        print(f"Test loader created with {len(self.test_loader)} batches")


    def _model_fit(self):
        
        if self.checkpointed:
            best_epoch = self.epoch
            losses = self.losses
            test_losses = self.test_losses
            best_model = copy.deepcopy(self.model.state_dict())
            early_stop_counter = self.early_stop_counter
            best_test_loss = min(test_losses)
        else:
            losses = []
            test_losses = []
            best_model = None
            early_stop_counter = 0
            best_test_loss = float('inf')
            best_epoch = 0

        start_time = time.time()
        
        max_norm = 0.2

        for epoch in range(len(losses), self.config['train']['num_epochs']):
            self.model.train()
            epoch_loss = 0

            ## --------------------------------------- Train ---------------------------------------
            for inputs, targets in self.train_loader: 
                inputs, targets = inputs, targets
                self.optimizer.zero_grad()
                total_loss = 0.0

                for n in range(targets.shape[1]):
                    # print(f"Step {n+1}/{targets.shape[1]}, VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    target = targets[:, n, :]  # shape: [B, input_dim]

                    # Forward pass
                    outputs = self.model(inputs)  # shape: [B, input_dim]
                    loss = self.criterion(outputs, target)

                    # Backward and optimization for current step only
                    total_loss += loss
                    loss.backward()

                    # Prepare input for next step
                    inputs = torch.cat((inputs[:, 1:, :], outputs.detach().unsqueeze(1)), dim=1)

                epoch_loss += total_loss.item() / targets.shape[1]

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.optimizer.step()

            losses.append(epoch_loss / len(self.train_loader))

            ## --------------------------------------- Test ---------------------------------------
            # Evaluate the model on the test set
            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    inputs, targets = inputs, targets
                    for n in range(targets.shape[1]):
                        target = targets[:, n, :]
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, target)
                        test_loss += loss.item() / targets.shape[1]
                        inputs = torch.cat((inputs[:, 1:, :], outputs.unsqueeze(1)), dim=1)

            test_losses.append(test_loss / len(self.test_loader))
            
            ## ------------------------------- Early stop and Checkpoint -------------------------------
            # Early stopping and saving the best model
            if epoch > 1:
                if np.isnan(test_losses[-1]) or np.isnan(losses[-1]):
                    print(f'NaN loss at epoch {epoch+1}. Stopping training.')
                    self.model.load_state_dict(best_model)
                    break
                elif test_loss / len(self.test_loader) < best_test_loss:
                    best_test_loss = test_loss / len(self.test_loader)
                    best_model = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch + 1
                    # print(f'Best model saved at epoch {best_epoch} with test loss: {best_test_loss:.4f}')
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= self.config['train']['patience']:
                        print(f'Early stopping at epoch {epoch+1}')
                        self.model.load_state_dict(best_model)
                        print(f'Best model loaded from epoch {best_epoch}, with test loss: {best_test_loss:.4f}')
                        break

                if (epoch + 1) % 5 == 0:
                    # Save model checkpoint every 5 epochs
                    # Save model losses, current weights, best weights, and optimizer state 

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'losses': losses,
                        'test_losses': test_losses,
                        'early_stop_counter': early_stop_counter,
                        'best_model': best_model
                    }, self.paths_bib.checkpoint_path)
                    # print(f"Checkpoint saved at epoch {epoch+1} to {self.paths_bib.checkpoint_path}")
                
            best_flag = 'X' if (epoch + 1) == best_epoch else ' '
            checkpoint_flag = 'X' if (epoch + 1) % 5 == 0 else ' '
            print(f"| Epoch: {epoch+1:<4}/{self.config['train']['num_epochs']:<4} | Train Loss: {losses[-1]:7.4f} | Test Loss: {test_losses[-1]:7.4f} | Best: {best_flag:<1} | Patience: {early_stop_counter:<3}/{self.config['train']['patience']} | Checkpoint: {checkpoint_flag:<1} |")

        end_time = time.time()
        print('\n\nTime taken for training: ', end_time - start_time)
        print('Time taken per epoch: ', (end_time - start_time) / (epoch + 1))

        # Save the final model after training
        torch.save(self.model.state_dict(), self.paths_bib.model_path)
        print(f"Final model saved to {self.paths_bib.model_path}")
        # Save the training and test losses
        with open(self.paths_bib.model_dir + 'losses.pkl', 'wb') as f:
            pickle.dump({'train_losses': losses, 'test_losses': test_losses}, f)

        print(f"Training and test losses saved to {self.paths_bib.model_dir + 'losses.pkl'}")

        print('\nTraining complete')


    def pred(self):
        print(f"{'#'*20}\t{'Predicting...':<20}\t{'#'*20}")

        time_lag = self.config['params']['time_lag']
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            if self.config['latent_type'] == 'dls':
                num_snaps = f['dof_u'].shape[0]
            elif self.config['latent_type'] == 'pod' or self.config['latent_type'] == 'bvae':
                num_snaps = f['dofs'].shape[0]

        for pred_name in self.config['predictions'].keys():
            print(f"Predicting for {pred_name}")

            pred_lim = self.config['predictions'][pred_name].get('lim', None)
            if pred_lim is None:
                pred_lim = np.inf

            if self.config['predictions'][pred_name]['init'] == 'val':
                idx = np.arange(self.val_indices[0], self.val_indices[0] + time_lag)
                if 'extrap' in self.config['predictions'][pred_name].get('arg', ''):
                    num_predictions = pred_lim
                else:
                    num_predictions = min(pred_lim, len(self.val_indices) - time_lag)

            elif self.config['predictions'][pred_name]['init'] == 'train':
                idx = np.arange(0, time_lag)
                if 'extrap' in self.config['predictions'][pred_name].get('arg', ''):
                    num_predictions = pred_lim
                else:
                    num_predictions = min(pred_lim, num_snaps - time_lag)

            

            # get initial sequence
            with h5py.File(self.paths_bib.latent_path, 'r') as f:
                if self.config['latent_type'] == 'dls':
                    dof_u = f['dof_u'][idx]
                    dof_v = f['dof_v'][idx]
                    initial_input = np.concatenate((dof_u, dof_v), axis=1)
                elif self.config['latent_type'] == 'pod':
                    initial_input = f['dofs'][idx, :self.config['latent_params']['num_modes'] ]
                elif self.config['latent_type'] == 'bvae':
                    initial_input = f['dofs'][idx, :self.config['latent_params']['latent_dim'] ]
            # print(f"Initial input shape: {initial_input.shape}")

            # make predictions
            pred = self._predict(initial_input, num_predictions=num_predictions)
        
            # split and save predictions
            print(f"Saving predictions to {self.paths_bib.predictions_dir + pred_name + '_pred.h5'}")
            if self.config['latent_type'] == 'dls':
                pred_dof_u = pred[:, :self.config['params']['input_dim'] // 2]
                pred_dof_v = pred[:, self.config['params']['input_dim'] // 2:]

                with h5py.File(self.paths_bib.predictions_dir + pred_name + '_pred.h5', 'w') as f:
                    f.create_dataset('dof_u', data=pred_dof_u)
                    f.create_dataset('dof_v', data=pred_dof_v)
                    f.create_dataset('idx', data=np.arange(idx[0], idx[0] + time_lag + num_predictions))
            elif self.config['latent_type'] == 'pod' or self.config['latent_type'] == 'bvae':
                
                with h5py.File(self.paths_bib.predictions_dir + pred_name + '_pred.h5', 'w') as f:
                    f.create_dataset('dofs', data=pred)
                    f.create_dataset('idx', data=np.arange(idx[0], idx[0] + time_lag + num_predictions))

            print(f"Saved!\n")

        # reconstruct the predictions
        print(f"\nReconstructing all predictions")
        self._pred_rec()


    def _predict(self, initial_input, num_predictions=1):
        """
        Predict the test set
        """
        
        self.model.eval()
        """
        Predict long-term future values using the model.

        Args:
            model: The trained model.
            initial_input: The initial input sequence.
            time_lag: The length of the input sequence.
            num_predictions: The number of future values to predict.

        Returns:
            predictions: The predicted future values.
        """
        # Normalize the initial input using the mean and std
        with open(os.path.join(self.paths_bib.model_dir, 'dof_scaler.pkl'), 'rb') as f:
            dof_mean, dof_std = pickle.load(f)
        if dof_mean.dtype == torch.float32:
            dof_mean = dof_mean.numpy()
        if dof_std.dtype == torch.float32:
            dof_std = dof_std.numpy()

        initial_input = (initial_input - dof_mean) / dof_std
        
        
        time_lag = self.config['params']['time_lag']
        
        if initial_input.shape[0] != 1:
            initial_input = initial_input[np.newaxis, ...]
        if initial_input.shape[1] != time_lag:
            print(f"Initial input shape {initial_input.shape[1]} does not match time lag {time_lag}, input will be truncated.")
            initial_input = initial_input[:, :time_lag, :]
        current_input = torch.tensor(initial_input, dtype=torch.float32).to(self.device)

        self.model.eval()
        predictions = []

        start_time = time.time()
        with torch.no_grad():
            interval = num_predictions // 10
            for n in range(num_predictions):
                if n % interval == 0:
                    print(f"Predicting step {n+1}/{num_predictions}")
                
                output = self.model(current_input)
                # print('Output shape: ', output[np.newaxis,:,:].shape)
                predictions.append(output.to('cpu').numpy())  # Ensure tensor is moved to CPU before converting to NumPy
                # Update the input for the next prediction
                current_input = torch.cat((current_input[:,1:, :], output.unsqueeze(0)), dim=1)
                # print('Current input shape: ', current_input.shape)

        end_time = time.time()
        print('Time taken for long-term prediction: ', end_time - start_time)
        print('Time taken per prediction: ', (end_time - start_time)/num_predictions)
        predictions = np.array(predictions)  # Convert list to NumPy array
        print(f"Predictions shape: {predictions.shape}")
        predictions = np.concatenate((initial_input[0,:,:], predictions.squeeze()), axis=0)  # Concatenate 

        # Denormalize the predictions
        predictions = (predictions * dof_std) + dof_mean

        return predictions
        

    def _pred_rec(self):

        # loop over all prediction sets in the predictions directory
        for pred_file in os.listdir(self.paths_bib.predictions_dir):
            if pred_file.endswith('.h5') and 'rec' not in pred_file and any(key in pred_file for key in self.config['predictions'].keys()):
                print(f"Reconstructing predictions from {pred_file}")
                rec_path = os.path.join(self.paths_bib.predictions_dir, 'rec_' + pred_file)
                pred_path = os.path.join(self.paths_bib.predictions_dir, pred_file)

                if self.config['latent_type'] == 'dls':
                    with h5py.File(pred_path, 'r') as f:
                        pred_dof_u = f['dof_u'][:]
                        pred_dof_v = f['dof_v'][:]
                        idx = f['idx'][:]

                    # Load the latent config
                    with open(self.paths_bib.latent_path.replace('.h5', '_config.pkl'), 'rb') as f:
                        latent_config = pickle.load(f)

                    dls.gfem_recon_long(config=latent_config,
                                        rec_path=rec_path,
                                        dof_u=pred_dof_u.T,
                                        dof_v=pred_dof_v.T,
                                        batch_size=1000)
                    
                    # copy idx field from original file to rec file
                    with h5py.File(rec_path, 'a') as f:
                        f.create_dataset('idx', data=idx)
                elif self.config['latent_type'] == 'pod':
                    with h5py.File(pred_path, 'r') as f:
                        pred_dofs = f['dofs'][:]
                        idx = f['idx'][:]

                    # Load the latent config
                    with open(self.paths_bib.latent_path.replace('.h5', '_config.pkl'), 'rb') as f:
                        latent_config = pickle.load(f)

                    pod.pod_recon_long(config=latent_config,
                                       dofs=pred_dofs,
                                       rec_path=rec_path,
                                       latent_path=self.paths_bib.latent_path,
                                       batch_size=1000)
                    
                    # copy idx field from original file to rec file
                    with h5py.File(rec_path, 'a') as f:
                        f.create_dataset('idx', data=idx)

                elif self.config['latent_type'] == 'bvae':
                    with h5py.File(pred_path, 'r') as f:
                        pred_dofs = f['dofs'][:]
                        idx = f['idx'][:]

                    # Load the latent config
                    with open(self.paths_bib.latent_path.replace('.h5', '_config.pkl'), 'rb') as f:
                        latent_config = pickle.load(f)

                    bvae_model = models.bvae_model(self.config['latent_params']['latent_dim'])
                    bvae_model.load_state_dict(torch.load(self.paths_bib.latent_model_path, weights_only=True))
                    bvae_model.to(self.device)

                    models.bvae_batch_decode(
                        model=bvae_model,
                        dofs=pred_dofs,
                        rec_path=rec_path,
                        data_path=self.paths_bib.data_path,
                        latent_path=self.paths_bib.latent_path,
                        device=self.device
                    )

                    with h5py.File(rec_path, 'a') as f:
                        f.create_dataset('idx', data=idx)
                

    def eval(self):
        """
        Evaluate the model
        """
        import lib.plotting as plots
        print(f"{'#'*20}\t{'Evaluating model...':<20}\t{'#'*20}")
        self.model.eval()
        nx = self.l_config.nx
        ny = self.l_config.ny
        nx_t = self.l_config.nx_t
        ny_t = self.l_config.ny_t

        # point probe info
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        x = x[:nx_t]
        y = y[:ny_t]
        X, Y = np.meshgrid(x, y)

        # find y closest  = 0.112
        y_closest = np.argmin(np.abs(y - 0.112))
        # print('y_closest: ', y[y_closest])

        # find x closest to 0.233 and 0.765
        x_closest1 = np.argmin(np.abs(x - 0.233))
        x_closest2 = np.argmin(np.abs(x - 0.765))
        # print('x_closest1: ', x[x_closest1])
        # print('x_closest2: ', x[x_closest2])

        point_1 = (x[x_closest1], y[y_closest])
        point_2 = (x[x_closest2], y[y_closest])
        point_1_idx = (x_closest1, y_closest)
        point_2_idx = (x_closest2, y_closest)
        
        time_lag = self.config['params']['time_lag']
        true_path = self.paths_bib.data_path
        self.compute_TKE_true(true_path)

        # Load the predictions
        for pred_file in os.listdir(self.paths_bib.predictions_dir):
            if pred_file.endswith('.h5') and 'rec' in pred_file and any(key in pred_file for key in self.config['predictions'].keys()):
                print(f"Loading predictions from {pred_file}")
                pred_name = pred_file.replace('rec_', '').replace('_pred.h5', '')
                self.paths_bib.pred_fig_dir = os.path.join(self.paths_bib.fig_dir, pred_name + '/')
                self.paths_bib.pred_metrics_dir = os.path.join(self.paths_bib.metrics_dir, pred_name + '/')
                
                os.makedirs(self.paths_bib.pred_fig_dir, exist_ok=True)
                os.makedirs(self.paths_bib.pred_metrics_dir, exist_ok=True)

                pred_path = os.path.join(self.paths_bib.predictions_dir, pred_file)
                
                with h5py.File(os.path.join(self.paths_bib.predictions_dir, pred_file), 'r') as f:
                    len_pred = f['Q_rec'].shape[0]
                    idx = f['idx'][:]
                    uv_rec_p1 = f['Q_rec'][:len_pred, point_1_idx[0], point_1_idx[1], :]
                    uv_rec_p2 = f['Q_rec'][:len_pred, point_2_idx[0], point_2_idx[1], :]

                with h5py.File(self.paths_bib.data_path, 'r') as f:
                    num_snaps = f['UV'].shape[0]
                    if idx[0] + time_lag + len_pred > num_snaps:
                        print(f"error metrics will be limited to t={idx[0]/100} to t={(num_snaps)/100}")
                        eval_len = num_snaps - idx[0]
                        eval_idx = idx[:eval_len]
                    else:
                        eval_len = len_pred
                        eval_idx = idx

                    if self.config['predictions'][pred_name].get('arg', '') == 'extrap':
                        true_idx = list(range(max(0, idx[0] - int(1*len_pred)), min(num_snaps, idx[0] + eval_len)))
                        eval_idx = eval_idx - true_idx[0] # adjust eval_idx to start from truth t=0
                    else:
                        true_idx = eval_idx

                    mean_uv_p1 = f['mean'][point_1_idx[0], point_1_idx[1], :]
                    mean_uv_p2 = f['mean'][point_2_idx[0], point_2_idx[1], :]
                    uv_p1 = f['UV'][true_idx, point_1_idx[0], point_1_idx[1], :] - mean_uv_p1[np.newaxis, ...]
                    uv_p2 = f['UV'][true_idx, point_2_idx[0], point_2_idx[1], :] - mean_uv_p2[np.newaxis, ...]

                point_dict = {
                    "truth": {
                        "p1": uv_p1,
                        "p2": uv_p2
                    },
                    "pred": {
                        "p1": uv_rec_p1,
                        "p2": uv_rec_p2
                    }
                }

                print('Loaded predictions and truth')

                self.compute_TKE(pred_path)
                self.compute_RMS(true_path, pred_path, eval_idx=eval_idx, batch_size=1000)

                # plot losses, RMS, TKE, Coherence
                print(f'\nGenerating plots, saving')
                plots.plot_loss(self)
                print('Loss plot done')
                plots.plot_rms(self, pred_path=pred_path, eval_idx=eval_idx, true_idx=true_idx)
                print('RMS plot done\n')
                plots.plot_tke(self, true_path=true_path, pred_path=pred_path, idx=idx, eval_idx=eval_idx, true_idx=true_idx)
                print('TKE plot done')
                plots.plot_PSDs(self, point_dict)
                print('PSD plot done')
                plots.plot_coherence(self, point_dict, eval_idx=eval_idx, true_idx=true_idx)
                print('Coherence plot done')
                plots.plot_points(self)
                print('Point plot done')
                plots.plot_point_data(self, point_dict, idx=idx, eval_idx=eval_idx, true_idx=true_idx)
                print('Point data plot done')
                if self.config['model'] == 'tr_enc':
                    plots.attention_maps(self)
                    print('Attention map plot done')
                plots.plot_phase_portraits(self, point_dict)
                plots.plot_spectrograms(self, point_dict, idx=idx, true_idx=true_idx)
                print('Spectrogram plot done')
                plots.coeff_PDF(self, point_dict, eval_idx=eval_idx, true_idx=true_idx)
                print('Coefficient PDF plot done\n\n')
                
    def compute_TKE(self, pred_path, batch_size=1000):
        """
        Compute the Turbulent Kinetic Energy (TKE) for predictions and respective ground truth data.
        The TKE is calculated as the sum of the squares of the velocity components at each time. 
        Data gets saved to predictions file as a field
        
        """
        import numpy as np
        import h5py

        with h5py.File(pred_path, 'r+') as f_pred:
            pred_shape = f_pred['Q_rec'].shape

            num_batches = pred_shape[0] // batch_size
            if pred_shape[0] % batch_size != 0:
                num_batches += 1
            tke_pred = np.zeros((pred_shape[0]), dtype=np.float32)
            
            if 'tke_pred' not in f_pred.keys() or any(x in self.config['overwrite'] for x in ['l', 'm', 'r']):
                print(f"Computing TKE for predictions in {pred_path}")
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, pred_shape[0])
                    if start_idx >= pred_shape[0]:
                        break
                    
                    # Get the predictions for the current batch
                    pred_batch = f_pred['Q_rec'][start_idx:end_idx]
                    
                    # Compute TKE for the current batch
                    tke_batch = 1/2 * np.sum(pred_batch**2, axis=(1, 2, 3))
                    tke_pred[start_idx:end_idx] = tke_batch

                    print(f"Computed TKE for batch {i+1}/{num_batches}, shape: {tke_batch.shape}, start_idx: {start_idx}, end_idx: {end_idx}")
                
                # Save the TKE to the predictions file
                if 'tke_pred' in f_pred.keys():
                    del f_pred['tke_pred']
                f_pred.create_dataset('tke_pred', data=tke_pred, dtype=np.float32)

        print(f"TKE computed and saved to pred_path in fields 'tke_pred'")


    def compute_TKE_true(self, true_path, batch_size=1000):
        import numpy as np
        import h5py
        with h5py.File(true_path, 'r+') as f_true:
            true_shape = f_true['UV'].shape

            num_batches = true_shape[0] // batch_size
            if true_shape[0] % batch_size != 0:
                num_batches += 1
            tke_true = np.zeros((true_shape[0]), dtype=np.float32)

            if self.paths_bib.latent_id + '_tke_true' not in f_true.keys():
                mean = f_true['mean'][:self.l_config.nx_t, :self.l_config.ny_t]
                print(f"Computing TKE for ground truth from {true_path}")
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, true_shape[0])
                    if start_idx >= true_shape[0]:
                        break
                    
                    # Get the ground truth data for the current batch
                    true_batch = f_true['UV'][start_idx:end_idx, :self.l_config.nx_t, :self.l_config.ny_t, :] - mean[np.newaxis, ...]
                    
                    # Compute TKE for the current batch
                    tke_batch = 1/2 * np.sum(true_batch**2, axis=(1, 2, 3))
                    tke_true[start_idx:end_idx] = tke_batch

                    print(f"Computed TKE for batch {i+1}/{num_batches}, shape: {tke_batch.shape}, start_idx: {start_idx}, end_idx: {end_idx}")
                    # Save the TKE to the predictions file
                if self.paths_bib.latent_id + '_tke_true' in f_true.keys():
                    del f_true[self.paths_bib.latent_id + '_tke_true']
                f_true.create_dataset(self.paths_bib.latent_id + '_tke_true', data=tke_true, dtype=np.float32)
        print(f"TKE computed and saved to true_path in fields '{self.paths_bib.latent_id + '_tke_true'}'")

    def compute_RMS(self, true_path, pred_path, eval_idx, batch_size=1000):
        """
        Compute the RMS error for predictions and respective ground truth data.
        The RMS error is calculated as the square root of the mean squared error at each time. 
        Data gets saved to predictions file as a field
        """
        import numpy as np
        import h5py

        time_lag = self.config['params']['time_lag']

        with h5py.File(pred_path, 'r+') as f_pred:
            pred_shape = f_pred['Q_rec'].shape
            
            num_snaps = len(eval_idx) - time_lag

            num_batches = len(eval_idx) // batch_size
            if num_snaps % batch_size != 0:
                num_batches += 1

            rms_pred = np.zeros((pred_shape[1], pred_shape[2], pred_shape[3]), dtype=np.float32)

            if 'rms_pred' not in f_pred.keys() or any(x in self.config['overwrite'] for x in ['l', 'm', 'r']):
                print(f"\nComputing RMS for predictions in {pred_path}")
                sum_squared_pred = np.zeros((pred_shape[1], pred_shape[2], pred_shape[3]), dtype=np.float32)
                for i in range(num_batches):
                    start_idx = time_lag + i * batch_size
                    end_idx = min(time_lag + (i + 1) * batch_size, len(eval_idx))
                    if start_idx >= pred_shape[0]:
                        break
                    
                    # Get the predictions for the current batch
                    pred_batch = f_pred['Q_rec'][start_idx:end_idx]
                    # Compute sum of squares of batch
                    sum_squared_pred += np.sum(pred_batch**2, axis=0)

                # Compute RMS for predictions
                rms_pred = np.sqrt(sum_squared_pred / (num_snaps))
                rms_pred = rms_pred.transpose((2,0,1))  
                # Save the RMS to the predictions file
                if 'rms_pred' in f_pred.keys():
                    del f_pred['rms_pred']
                f_pred.create_dataset('rms_pred', data=rms_pred, dtype=np.float32)

            with h5py.File(true_path, 'r+') as f_true:
                true_shape = f_true['UV'].shape
                

                num_batches = (len(eval_idx) - time_lag) // batch_size
                if num_snaps % batch_size != 0:
                    num_batches += 1

                rms_true =np.zeros((pred_shape[1], pred_shape[2], pred_shape[3]), dtype=np.float32)
                if 'rms_true' not in f_pred.keys() or any(x in self.config['overwrite'] for x in ['l', 'm', 'r']):
                    mean = f_true['mean'][:pred_shape[1], :pred_shape[2]]
                    print(f"Computing RMS for ground truth from {true_path}")
                    sum_squared_true = np.zeros((pred_shape[1], pred_shape[2], pred_shape[3]), dtype=np.float32)
                    for i in range(num_batches):
                        start_idx = eval_idx[0] + time_lag + i * batch_size
                        end_idx = min(eval_idx[0] + time_lag + (i + 1) * batch_size, eval_idx[-1] )
                        if start_idx >= true_shape[0]:
                            print('Breaking out of loop, start_idx >= true_shape[0]')
                            break

                        
                        # Get the ground truth data for the current batch
                        true_batch = f_true['UV'][start_idx:end_idx, :pred_shape[1], :pred_shape[2], :] - mean[np.newaxis, ...]
                        
                        # Compute sum of squares of batch
                        sum_squared_true += np.sum(true_batch**2, axis=0)

                    # Compute RMS for ground truth
                    rms_true = np.sqrt(sum_squared_true / (len(eval_idx) - time_lag))
                    rms_true = rms_true.transpose((2,0,1))
                    # Save the RMS to the predictions file
                    if 'rms_true' in f_pred.keys():
                        del f_pred['rms_true']
                    f_pred.create_dataset('rms_true', data=rms_true, dtype=np.float32)
        print(f"RMS computed and saved to pred_path in fields 'rms_pred' and 'rms_true'")