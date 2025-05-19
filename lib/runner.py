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
        """
        Initialize the runner 
        """
        super(runner, self).__init__()
        self.config = config
        self.device = config['device']
        
        is_init_path, self.paths_bib = init.init_path(config)

        # set output and error streams to a log file
        sys.stdout = open(self.paths_bib.log_path, 'w')
        sys.stderr = open(self.paths_bib.log_path, 'a')
        print(f'Using device: {self.device}')

        # Print the keys and nested keys in the config file
        print(f"{'#'*20}\t{'Configuration':<20}\t{'#'*20}")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"{key}")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        print(f"{'#'*20}\t{'Pathing initialization':<20}\t{'#'*20}")
        
        print(f"Paths initialized")
        # print directory structure
        if is_init_path:
            DisplayTree(header=True, ignoreList = ['*.pyc'])


        # get model info
        self.get_data()
        self.get_model()
        self.compile_model()
        



    def get_data(self):
        """
        Load the latent coefficients
        """
        print(f"{'#'*20}\t{'Loading data...':<20}\t{'#'*20}")
        
        if not os.path.exists(self.paths_bib.latent_path) or self.config['overwrite'] == 'l':
            # computing the latent coefficients
            print(f"\nCoefficients not found. Computing latent coefficients")
            with h5py.File(self.paths_bib.data_path, 'r') as f:
                num_snaps = f['UV'].shape[0]
            latent_config = dls.gfem_2d_long(data_path = self.paths_bib.data_path, 
                                            field_name = 'UV',
                                            latent_file = self.paths_bib.latent_path, 
                                            patch_size = self.config['latent_params']['patch_size'],
                                            num_modes = self.config['latent_params']['num_modes'],
                                            batch_size = num_snaps // 20)
            # save latent config
            with open(self.paths_bib.latent_path.replace('.h5', '_config.pkl'), 'wb') as f:
                pickle.dump(latent_config, f)
            print(f"Latent coefficient config saved")

        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            # Define the model input dimension, = total dofs
            self.config['params']['input_dim'] = 2 * f['dof_u'].shape[1]

            total_snaps = f['dof_u'].shape[0]
            print(f"Total snaps: {total_snaps}")

            # find indices for train, test, and validation sets
            train_indices = np.arange(0, int(total_snaps * self.config['train']['train_split']))
            test_indices = np.arange(len(train_indices), len(train_indices) + int(len(train_indices) *self.config['train']['test_split']))
            val_indices = np.arange(len(train_indices) + len(test_indices), total_snaps)
            
            # Print a table with the set info
            sample_train = self.config['train']['sample_train']
            sample_test = self.config['train']['sample_test']

            print(f"{'Set':<12}|{'Total':<10}|{'First Idx':<12}|{'Last Idx':<12}|{'Sampled':<10}")
            print("-" * 56)
            print(f"{'Train':<12}|{len(train_indices):<10}|{train_indices[0]:<12}|{train_indices[-1]:<12}|{sample_train:<10}")
            print(f"{'Test':<12}|{len(test_indices):<10}|{test_indices[0]:<12}|{test_indices[-1]:<12}|{sample_test:<10}")
            print(f"{'Validation':<12}|{len(val_indices):<10}|{val_indices[0]:<12}|{val_indices[-1]:<12}{'-':<10}")

            self.train_indices = np.sort(datas.sample_series_indices(
                                        len(train_indices), 
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
 

    def get_model(self):
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
            print(f"Loading model weights from {self.paths_bib.model_path}")
            self.model.load_state_dict(torch.load(self.paths_bib.model_path, weights_only=True))

        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")


    def compile_model(self):
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
            checkpoint = torch.load(self.paths_bib.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_loader = checkpoint['train_loader']
            self.test_loader = checkpoint['test_loader']
            self.criterion = checkpoint['criterion']
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
            self.config['mode'] = 'eval'
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
        if self.checkpointed:
            print(f"Checkpointed model loaded. Continuing training from epoch {self.epoch}")
        else:
            self.get_train_data()
        self.model_fit()
        

    def get_train_data(self):
        """
        Get training and test data
        """
        print('Getting training and test data')
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            dof_u = f['dof_u']
            dof_v = f['dof_v']

            dof_u_train = np.array([dof_u[idx:idx+1,:] for idx in self.train_indices]).squeeze()
            dof_v_train = np.array([dof_v[idx:idx+1,:] for idx in self.train_indices]).squeeze()

            dof = np.concatenate((dof_u_train, dof_v_train), axis=1)

            dof_mean = np.mean(dof, axis=0)
            dof_std = np.std(dof, axis=0)

            tl = self.config['params']['time_lag']
            ta = self.config['train']['train_ahead']

            # Prepare arrays for X_train and Y_train
            dof_dim = self.config['params']['input_dim']

            X_train = np.empty((len(self.train_indices), tl, dof_dim), dtype=np.float32)
            Y_train = np.empty((len(self.train_indices), ta, dof_dim), dtype=np.float32)
            X_test = np.empty((len(self.test_indices), tl, dof_dim), dtype=np.float32)
            Y_test = np.empty((len(self.test_indices), ta, dof_dim), dtype=np.float32)

            # Fill X_train and Y_train
            for i, idx in enumerate(self.train_indices):
                dof = np.concatenate((dof_u[idx:idx+tl+ta], dof_v[idx:idx+tl+ta]), axis=1)
                dof = datas.normalize_data(dof, mean=dof_mean, std=dof_std)
                X_train[i] = dof[:tl]
                Y_train[i] = dof[tl:tl+ta]
                if i % 500 == 0:
                    print(f"Got train data {i}/{len(self.train_indices)}")
            print('Got train data')

            # Fill X_test and Y_test
            for i, idx in enumerate(self.test_indices):
                dof = np.concatenate((dof_u[idx:idx+tl+ta], dof_v[idx:idx+tl+ta]), axis=1)
                dof = datas.normalize_data(dof, mean=dof_mean, std=dof_std)
                X_test[i] = dof[:tl]
                Y_test[i] = dof[tl:tl+ta]
                if i % 100 == 0:
                    print(f"Got test data {i}/{len(self.test_indices)}")
            print('Got test data')

        print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

        # convert to torch tensors and data loader
        self.train_loader = datas.make_dataloader(X_train, Y_train, batch_size=self.config['train']['batch_size'], shuffle=True)
        self.test_loader = datas.make_dataloader(X_test, Y_test, batch_size=self.config['train']['batch_size'], shuffle=False)
    
    
    def model_fit(self):
        
        
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
            print(f"Epoch [{epoch+1}/{self.config['train']['num_epochs']}], Train Loss: {losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")


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
                    print(f'Best model saved at epoch {best_epoch} with test loss: {best_test_loss:.4f}')
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= self.config['train']['patience']:
                        print(f'Early stopping at epoch {epoch+1}')
                        self.model.load_state_dict(best_model)
                        print(f'Best model loaded from epoch {best_epoch}, with test loss: {best_test_loss:.4f}')
                        break

                if epoch % 5 == 0:
                    # Save model checkpoint every 5 epochs
                    # Save model losses, current weights, best weights, and optimizer state
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'losses': losses,
                        'test_losses': test_losses,
                        'early_stop_counter': early_stop_counter,
                        'best_model': best_model,
                        'train_loader': self.train_loader,
                        'test_loader': self.test_loader,
                    }, self.paths_bib.checkpoint_path)
                    print(f"Checkpoint saved at epoch {epoch+1} to {self.paths_bib.checkpoint_path}")
                    

        end_time = time.time()
        print('Time taken for training: ', end_time - start_time)
        print('Time taken per epoch: ', (end_time - start_time) / epoch)

        # Save the final model after training
        torch.save(self.model.state_dict(), self.paths_bib.model_path)
        print(f"Final model saved to {self.paths_bib.model_path}")
        # Save the training and test losses
        with open(self.paths_bib.model_dir + 'losses.pkl', 'wb') as f:
            pickle.dump({'train_losses': losses, 'test_losses': test_losses}, f)