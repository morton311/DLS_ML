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
import lib.transformer as tr
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
        sys.stdout = open(paths.log_path, 'w')
        sys.stderr = open(paths.log_path, 'a')

        print(f'Using device: {self.device}')
        if is_init_path:
            DisplayTree(header=True, ignoreList=['*.pyc', '*.png'], maxDepth=5)
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


    def _latent_split(self):
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            input_dim = 2 * f['dof_u'].shape[1]
            self.config['params']['input_dim'] = input_dim
            total_snaps = f['dof_u'].shape[0]

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
        self.model = tr.TransformerEncoderModel(
                    time_lag=self.config['params']['time_lag'],
                    input_dim=self.config['params']['input_dim'],
                    d_model=self.config['params']['d_model'],
                    nhead=self.config['params']['nhead'],
                    num_layers=self.config['params']['num_layers']
                    ).to(self.device)
        
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

        # if checkpoint file exists, model doesn't exist, and overwrite is not set to 'l' or 'm', load the checkpoint
        check_flag = os.path.exists(self.paths_bib.checkpoint_path)
        model_flag = os.path.exists(self.paths_bib.model_path)
        if check_flag and not model_flag and not self.config['overwrite'] in ['l', 'm']: 
            print(f"Loading checkpoint from {self.paths_bib.checkpoint_path}")
            checkpoint = torch.load(self.paths_bib.checkpoint_path, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.losses = checkpoint['losses']
            self.test_losses = checkpoint['test_losses']
            self.early_stop_counter = checkpoint['early_stop_counter']
            print(f"Checkpoint loaded")
            self.checkpointed = True
            self.config['mode'] = 'train'

        # if model exists and overwrite is not set to 'l' or 'm', load the model and skip training
        elif model_flag and not self.config['overwrite'] in ['l', 'm']:
            print(f"Model already exists at {self.paths_bib.model_path}. Skipping training.")
            print(f"Loading model weights from {self.paths_bib.model_path}")
            self.model.load_state_dict(torch.load(self.paths_bib.model_path, weights_only=True))
            self.checkpointed = False
        else:
            print(f"Model does not exist at {self.paths_bib.model_path}. Training from scratch.")
            self.checkpointed = False
            self.config['mode'] = 'train'


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
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            dof_u = f['dof_u']
            dof_v = f['dof_v']

            tl = self.config['params']['time_lag']
            ta = self.config['train']['train_ahead']
            dof_dim = self.config['params']['input_dim']

            # Compute mean and std using only the training indices, in chunks to save memory
            dofs = torch.zeros(len(self.train_indices), dof_dim)
            for i, idx in enumerate(self.train_indices):
                u = torch.from_numpy(dof_u[idx:idx+1]).float()
                v = torch.from_numpy(dof_v[idx:idx+1]).float()
                dofs[i] = torch.cat((u, v), dim=1)
            dof_mean = torch.mean(dofs, dim=0)
            dof_std = torch.std(dofs, dim=0)

            self.dof_mean = dof_mean
            self.dof_std = dof_std

            with open(os.path.join(self.paths_bib.model_dir, 'dof_scaler.pkl'), 'wb') as f:
                pickle.dump((dof_mean, dof_std), f)

            # Helper to get normalized dof sequence as torch tensor
            def get_dof_seq(idx, length):
                u = torch.from_numpy(dof_u[idx:idx+length]).float()
                v = torch.from_numpy(dof_v[idx:idx+length]).float()
                dof = torch.cat((u, v), dim=1)
                dof = (dof - dof_mean) / dof_std
                return dof

            # Prepare lists for X/Y, then stack at the end
            X_train, Y_train = torch.zeros(len(self.train_indices), tl, dof_dim), torch.zeros(len(self.train_indices), ta, dof_dim)
            for i, idx in enumerate(self.train_indices):
                dof_seq = get_dof_seq(idx, tl + ta)
                X_train[i] = dof_seq[:tl]
                Y_train[i] = dof_seq[tl:tl+ta]
                if i % 500 == 0:
                    print(f"Got train data {i}/{len(self.train_indices)}")
            print('Got train data')

            X_test, Y_test = torch.zeros(len(self.test_indices), tl, dof_dim), torch.zeros(len(self.test_indices), ta, dof_dim)
            for i, idx in enumerate(self.test_indices):
                dof_seq = get_dof_seq(idx, tl + ta)
                X_test[i] = dof_seq[:tl]
                Y_test[i] = dof_seq[tl:tl+ta]
                if i % 100 == 0:
                    print(f"Got test data {i}/{len(self.test_indices)}")
            print('Got test data')


        print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}, dtype: {X_train.dtype}")
        print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}, dtype: {X_test.dtype}")

        # convert to data loader
        self.train_loader = datas.make_dataloader(X_train, Y_train, batch_size=self.config['train']['batch_size'], shuffle=True)
        print(f"Train loader created with {len(self.train_loader)} batches")
        self.test_loader = datas.make_dataloader(X_test, Y_test, batch_size=self.config['train']['batch_size'], shuffle=False)
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

        start_time = time.time()
        
        max_norm = 0.2

        for epoch in range(len(losses), self.config['train']['num_epochs']):
            self.model.train()
            epoch_loss = 0

            ## --------------------------------------- Train ---------------------------------------
            for inputs, targets in self.train_loader: 
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                total_loss = 0.0

                for n in range(targets.shape[1]):
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
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    for n in range(targets.shape[1]):
                        target = targets[:, n, :]
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, target)
                        test_loss += loss.item() / targets.shape[1]
                        inputs = torch.cat((inputs[:, 1:, :], outputs.unsqueeze(1)), dim=1)

            test_losses.append(test_loss / len(self.test_loader))
            
            ## ------------------------------- Early stop and Checkpoint -------------------------------
            # Early stopping and saving the best model
            if epoch > 0:
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


    def pred_old(self):
        print(f"{'#'*20}\t{'Predicting...':<20}\t{'#'*20}")
        # get initial sequence
        time_lag = self.config['params']['time_lag']
        if 'pred_lim' in self.config.keys():
            pred_lim = self.config['pred_lim']
        else:
            pred_lim = np.inf
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            dof_u = f['dof_u'][self.val_indices[0]:self.val_indices[0] + time_lag]
            dof_v = f['dof_v'][self.val_indices[0]:self.val_indices[0] + time_lag]
        initial_input = np.concatenate((dof_u, dof_v), axis=1)

        print(f"Initial input shape: {initial_input.shape}")

        num_predictions = len(self.val_indices) - time_lag
        if num_predictions > pred_lim:
            num_predictions = pred_lim

        # make predictions
        val_pred = self._predict(initial_input, num_predictions = num_predictions)
        
        # split and save predictions
        pred_dof_u = val_pred[:, :self.config['params']['input_dim'] // 2]
        pred_dof_v = val_pred[:, self.config['params']['input_dim'] // 2:]

        print('Saving predictions to file')
        with h5py.File(self.paths_bib.predictions_dir + 'val_pred.h5', 'w') as f:
            f.create_dataset('dof_u', data=pred_dof_u)
            f.create_dataset('dof_v', data=pred_dof_v)

        print(f"Predictions saved to {self.paths_bib.predictions_dir + 'val_pred.h5'}")

        # reconstruct the predictions
        print(f"Reconstructing predictions")
        self._pred_rec()

    def pred(self):
        print(f"{'#'*20}\t{'Predicting...':<20}\t{'#'*20}")

        time_lag = self.config['params']['time_lag']
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            num_snaps = f['dof_u'].shape[0]

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
                dof_u = f['dof_u'][idx]
                dof_v = f['dof_v'][idx]
            initial_input = np.concatenate((dof_u, dof_v), axis=1)
            # print(f"Initial input shape: {initial_input.shape}")

            # make predictions
            pred = self._predict(initial_input, num_predictions=num_predictions)
        
            # split and save predictions
            pred_dof_u = pred[:, :self.config['params']['input_dim'] // 2]
            pred_dof_v = pred[:, self.config['params']['input_dim'] // 2:]

            print(f"Saving predictions to {self.paths_bib.predictions_dir + pred_name + '_pred.h5'}")
            with h5py.File(self.paths_bib.predictions_dir + pred_name + '_pred.h5', 'w') as f:
                f.create_dataset('dof_u', data=pred_dof_u)
                f.create_dataset('dof_v', data=pred_dof_v)
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
                

    def eval(self):
        """
        Evaluate the model
        """
        import lib.plotting as plots
        print(f"{'#'*20}\t{'Evaluating model...':<20}\t{'#'*20}")
        
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

        # load indices of validation set
        # with open(self.paths_bib.model_dir + 'split_ids.pkl', 'rb') as f:
        #     indices = pickle.load(f)
        #     self.val_indices = indices['val_indices']

        eval_lim = 5064

        # Load the predictions
        for pred_file in os.listdir(self.paths_bib.predictions_dir):
            if pred_file.endswith('.h5') and 'rec' in pred_file and any(key in pred_file for key in self.config['predictions'].keys()):
                print(f"Loading predictions from {pred_file}")
                pred_name = pred_file.replace('rec_', '').replace('_pred.h5', '')
                self.paths_bib.pred_fig_dir = os.path.join(self.paths_bib.fig_dir, pred_name + '/')
                self.paths_bib.pred_metrics_dir = os.path.join(self.paths_bib.metrics_dir, pred_name + '/')
                
                os.makedirs(self.paths_bib.pred_fig_dir, exist_ok=True)
                os.makedirs(self.paths_bib.pred_metrics_dir, exist_ok=True)

                with h5py.File(os.path.join(self.paths_bib.predictions_dir, pred_file), 'r') as f:
                    len_pred = f['Q_rec'].shape[0]
                    if len_pred > eval_lim:
                        print(f"Limiting predictions to {eval_lim} samples")
                        len_pred = eval_lim
                    pred = f['Q_rec'][:len_pred]
                    uv_rec_p1 = pred[:len_pred, point_1_idx[0], point_1_idx[1], :]
                    uv_rec_p2 = pred[:len_pred, point_2_idx[0], point_2_idx[1], :]
                    idx = f['idx'][:len_pred]
                
                print(f'Loading truth from {self.paths_bib.data_path}')
                with h5py.File(self.paths_bib.data_path, 'r') as f:
                    num_snaps = f['UV'].shape[0]
                    if idx[0] + time_lag + len_pred > num_snaps:
                        print(f"error metrics will be limited to t={idx[0]/100} to t={(num_snaps)/100}")
                        eval_len = num_snaps - idx[0]
                        eval_idx = idx[:eval_len]
                    else:
                        eval_len = len_pred
                        eval_idx = idx

                    if idx[0] < 3000 and len_pred <= eval_lim and self.config['predictions'][pred_name].get('arg', '') == 'extrap':
                        true_idx = list(range(0, min(num_snaps, idx[0] + eval_len)))
                    else:
                        true_idx = eval_idx
                    mean = f['mean'][:nx_t, :ny_t]
                    truth = f['UV'][true_idx, :nx_t, :ny_t, :] - mean[np.newaxis, ...]
                    mean_uv_p1 = f['mean'][point_1_idx[0], point_1_idx[1], :]
                    mean_uv_p2 = f['mean'][point_2_idx[0], point_2_idx[1], :]
                    uv_p1 = f['UV'][true_idx, point_1_idx[0], point_1_idx[1], :] - mean_uv_p1[np.newaxis, ...]
                    uv_p2 = f['UV'][true_idx, point_2_idx[0], point_2_idx[1], :] - mean_uv_p2[np.newaxis, ...]

                print(f"Truth shape: {truth.shape}, Pred shape: {pred.shape}")

                print(eval_idx[0], eval_idx[-1])

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

                # plot losses, RMS, TKE, Coherence
                print(f'\nGenerating plots, saving')
                plots.plot_loss(self)
                print('Loss plot done')
                plots.plot_rms(self, truth=truth, pred=pred, eval_idx=eval_idx, true_idx=true_idx)
                print('RMS plot done\n')
                plots.plot_tke(self, truth=truth, pred=pred, idx=idx, eval_idx=eval_idx, true_idx=true_idx)
                print('TKE plot done')
                plots.plot_PSDs(self, point_dict)
                print('PSD plot done')
                plots.plot_coherence(self, point_dict, eval_idx=eval_idx, true_idx=true_idx)
                print('Coherence plot done')
                plots.plot_points(self)
                print('Point plot done')
                plots.plot_point_data(self, point_dict, idx=idx, eval_idx=eval_idx, true_idx=true_idx)
                print('Point data plot done')
                plots.attention_maps(self)
                print('Attention map plot done\n\n')
                

