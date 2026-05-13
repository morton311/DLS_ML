import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from datetime import datetime
import os
import numpy as np
import time
from tqdm import tqdm
import copy
import pickle
import h5py
from . import datas

def compute_bvae_latent(runner):
    with h5py.File(runner.paths_bib.data_path, 'r') as f:
        data = f['UV'][0]
        data_shape = data.transpose(2, 0, 1).shape
    bvae = bvae_model(data_shape, runner.config)
    bvae.to(runner.device)

    if not os.path.exists(runner.paths_bib.latent_model_path) and not runner.config['overwrite'] in ['l', 'm']:

        train_snaps = runner.config['latent_params'].get('train_snaps', 2500)
        train_split = runner.config['latent_params'].get('train_test_split', 0.8)
        test_split = runner.config['latent_params'].get('train_val_split', 0.1)

        
        with h5py.File(runner.paths_bib.data_path, 'r') as f:
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
            with open(runner.paths_bib.latent_dir + 'latent_scaler.pkl', 'wb') as f:
                pickle.dump((train_mean, train_std), f)

            train_set = datas.normalize_data(train_set, train_mean, train_std)
            test_set = datas.normalize_data(test_set, train_mean, train_std)

        print(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")
        print(f"Train set mean: {np.mean(train_set)}, Train set std: {np.std(train_set)}")
        print(f"Train set min: {np.min(train_set)}, Train set max: {np.max(train_set)}")

        train_set = train_set.transpose(0, 3, 1, 2)  # [S, C, H, W]
        test_set = test_set.transpose(0, 3, 1, 2)    # [S, C, H, W]

        # make train and test data loaders
        train_loader = datas.make_dataloader(
            torch.from_numpy(train_set).float().to(runner.device),
            torch.from_numpy(train_set).float().to(runner.device),
            batch_size=runner.config['latent_params'].get('batch_size', 256),
            shuffle=True
        )

        test_loader = datas.make_dataloader(
            torch.from_numpy(test_set).float().to(runner.device),
            torch.from_numpy(test_set).float().to(runner.device),
            batch_size=runner.config['latent_params'].get('batch_size', 256),
            shuffle=False
        )
        

        # create optimizer
        optimizer = torch.optim.Adam(bvae.parameters(), lr=runner.config['latent_params'].get('lr', 2e-4))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                    max_lr=runner.config['latent_params'].get('lr', 2e-4),
                                    total_steps=runner.config['latent_params'].get('num_epochs', 1000),
                                    div_factor=2, 
                                    final_div_factor=runner.config['latent_params'].get('lr', 2e-4)
                                                    /runner.config['latent_params'].get('lr_end', 1e-5),
                                    pct_start=0.2)
        beta_scheduler = betaScheduler(runner.config['latent_params']['beta'])

        bvae, losses = train_bvae(
            model=bvae.to(runner.device),
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            config=runner.config,
            scheduler=scheduler,
            beta_scheduler=beta_scheduler
        )

        # Save model and losses
        torch.save(bvae.state_dict(), runner.paths_bib.latent_model_path)
        with open(runner.paths_bib.latent_dir + 'bvae_losses.pkl', 'wb') as f:
            pickle.dump(losses, f)
        
        print(f"Latent model saved to {runner.paths_bib.latent_model_path}")
        print(f"Latent losses saved to {runner.paths_bib.latent_dir + 'bvae_losses.pkl'}")

    else:
        print(f"Latent model already exists at {runner.paths_bib.latent_model_path}. Loading model.")
        
        bvae.load_state_dict(torch.load(runner.paths_bib.latent_model_path, weights_only=True, map_location=runner.device))
        bvae.to(runner.device)


    if os.path.exists(runner.paths_bib.latent_path):
        print(f"Latent coefficients already exist at {runner.paths_bib.latent_path}. Skipping encoding.")
    else:
        # Encode the full dataset to get latent coefficients
        bvae_batch_encode(
            model=bvae,
            data_path=runner.paths_bib.data_path,
            latent_path=runner.paths_bib.latent_path,
            config=runner.config,
            device=runner.device
        )


class bvae_encoder(nn.Module):
    def __init__(self, data_shape, config):
        super().__init__()
        self.conv = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.pad = nn.ModuleList([])
        self.width = [ data_shape[1] ]
        self.height = [ data_shape[2] ]

        for i, filters in enumerate(config['latent_params']['filters']):
            if i == 0:
                in_channels = data_shape[0]
            else:
                in_channels = config['latent_params']['filters'][i-1]
            out_channels = filters
            self.conv.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1))
            self.act.append(nn.ELU())
            width = np.floor((self.width[-1] - 1 ) / (2) + 1)
            height = np.floor((self.height[-1] - 1) / (2) + 1)

            if (width % 2 == 0) and (height % 2 == 0):
                self.pad.append(nn.Identity())
            elif width % 2 == 0 and height % 2 != 0:
                self.pad.append(nn.ConstantPad2d((0,1,0,0), 0))
                height += 1
            elif width % 2 != 0 and height % 2 == 0:
                self.pad.append(nn.ConstantPad2d((0,0,0,1), 0))
                width += 1
            else:
                self.pad.append(nn.ConstantPad2d((0,1,0,1), 0))
                width += 1
                height += 1

            self.width.append(width)
            self.height.append(height)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(int(self.width[-1]*self.height[-1]*config['latent_params']['filters'][-1]), config['latent_params']['linear'][0])
        self.act7 = nn.ELU()
        self.out = nn.Linear(config['latent_params']['linear'][0], config['latent_params']['latent_dim'] * 2) 

        self.reshape_dim = [config['latent_params']['filters'][-1], int(self.width[-1]), int(self.height[-1])]

        

        print(self.width)
        print(self.height)
        for layer in self.pad:
            print(type(layer))

    def forward(self, x):
        # print('Encoder arch')
        # print(x.shape)
        for conv, act, pad in zip(self.conv, self.act, self.pad):
            x = act(conv(pad(x)))
            # print(x.shape)
            # print(type(pad))
        x = self.flatten(x)
        x = self.act7(self.fc1(x))
        x = self.out(x)
        return x
    
class bvae_decoder(nn.Module):
    def __init__(self, data_shape, config, encoder):
        super().__init__()
        self.deconv = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.pad = nn.ModuleList([])
        print(encoder.reshape_dim)
        self.width = [encoder.reshape_dim[1]]
        self.height = [encoder.reshape_dim[2]]
        
        for layer in reversed(encoder.pad):
            if type(layer) == nn.Identity:
                self.pad.append(nn.Identity())
            else:
                padding = tuple([-1*x for x in layer.padding])
                print(layer.padding , padding)
                self.pad.append(nn.ConstantPad2d(padding, 0))

        self.input = nn.Linear(config['latent_params']['latent_dim'], config['latent_params']['linear'][0])
        self.act1 = nn.ELU()
        self.fc2 = nn.Linear(config['latent_params']['linear'][0], int(math.prod(encoder.reshape_dim)))
        self.act2 = nn.ELU()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=tuple(encoder.reshape_dim))
        for i, filters in enumerate(reversed(config['latent_params']['filters'])):
            if i == len(config['latent_params']['filters']) - 1:
                out_channels = data_shape[0]
            else:
                out_channels = config['latent_params']['filters'][-(i+2)]
            in_channels = filters
            self.deconv.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))

            

            if i < len(config['latent_params']['filters']) - 1:
                self.act.append(nn.ELU())
            else:
                self.act.append(nn.Identity())

    def forward(self, x):
        # print('Decoder arch')
        x = self.act1(self.input(x))
        x = self.act2(self.fc2(x))
        x = self.unflatten(x)
        # print(x.shape)
        for deconv, act, pad in zip(self.deconv, self.act, self.pad):
            x = act(deconv(pad(x)))
            # print(x.shape)
            # print(type(pad))
        return x

class bvae_model(nn.Module):
    """
    A base class for a Bayesian Variational Autoencoder (BVAE) model.
    Convolutional encoder decoder model with reparameterization trick.
    """
    def __init__(self, data_shape, config):
        super().__init__()
        self.encoder = self.buildEncoder(data_shape, config)
        self.decoder = self.buildDecoder(data_shape, config, self.encoder)

    def buildEncoder(self, data_shape, config):
        encoder = bvae_encoder(data_shape, config)
        return encoder

    def buildDecoder(self, data_shape, config, encoder):
        decoder = bvae_decoder(data_shape, config, encoder)
        return decoder

    
    def sample(self, mean, logvariance):
        """
        Reparameterization trick 
        """

        std = torch.exp(0.5 * logvariance)
        epsilon = torch.rand_like(std)

        return mean + epsilon*std

    def forward(self, data):

        mean_logvariance = self.encoder(data)

        mean, logvariance = torch.chunk(mean_logvariance, 2, dim=1)

        z = self.sample(mean, logvariance)

        reconstruction = self.decoder(z)

        return reconstruction, mean, logvariance
    
def bvae_loss(reconstruction, data, mean, logvariance, beta):
    MSELoss = nn.MSELoss(reduction='mean').cuda()
    MSE = MSELoss(reconstruction, data)

    KLD = -0.5 * torch.mean(1 + logvariance - mean.pow(2) - logvariance.exp())

    loss = MSE + KLD * beta

    return loss, MSE, KLD

class betaScheduler:
    """Schedule beta, linear growth to max value"""

    def __init__(self, endvalue, startvalue=None, warmup=20):
        if startvalue is None:
            startvalue = endvalue / 5
        self.startvalue = startvalue
        self.endvalue = endvalue
        self.warmup = warmup

    def getBeta(self, epoch, prints=False):

        if epoch < self.warmup:
            beta = self.startvalue + (self.endvalue - self.startvalue) * epoch / self.warmup
            if prints:
                print(beta)
            return beta
        else:
            return self.endvalue

def train_bvae(model, train_loader, test_loader, optimizer, config, scheduler=None, beta_scheduler=None):

    best_test_loss = float('inf')
    early_stop_counter = 0
    losses = []
    test_losses = []
    patience = config['train']['patience']
    num_epochs = config['train']['num_epochs']
    beta = config['latent_params']['beta']

    lr_now = 0

    # Training loop
    start_time = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        logVar_batch = []

        ## --------------------------------------- Train ---------------------------------------
        for inputs, targets in train_loader: 
            optimizer.zero_grad()

            # Forward pass
            outputs, mean, logvariance = model(inputs)
            loss, MSE, KLD = bvae_loss(outputs, inputs, mean, logvariance, beta)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
            logVar_batch.append(np.exp(0.5* np.mean(logvariance.detach().cpu().numpy(), 0)))
        losses.append(epoch_loss / len(train_loader))

        if scheduler is not None:
            scheduler.step()
            lr_now = scheduler.get_last_lr()

        if beta_scheduler is not None:
            beta = beta_scheduler.getBeta(epoch)
        ## --------------------------------------- Test ---------------------------------------
        # Evaluate the model on the test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs, mean, logvariance = model(inputs)
                loss, MSE, KLD = bvae_loss(outputs, inputs, mean, logvariance, beta)
                test_loss += loss.item()
        test_losses.append(test_loss / len(test_loader))
        
        ## ------------------------------- Early stop and Checkpoint -------------------------------
        # Early stopping and saving the best model
        if epoch > 0:
            if np.isnan(test_losses[-1]) or np.isnan(losses[-1]):
                print(f'NaN loss at epoch {epoch+1}. Stopping training.')
                model.load_state_dict(best_model)
                break
            elif test_loss / len(test_loader) < best_test_loss:
                best_test_loss = test_loss / len(test_loader)
                best_model = copy.deepcopy(model.state_dict())

                best_epoch = epoch + 1
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    model.load_state_dict(best_model)
                    print(f'Best model loaded from epoch {best_epoch}, with test loss: {best_test_loss:.4f}')
                    break

        best_flag = 'X' if (epoch + 1) == best_epoch else ' '
        mode_collapse = (np.mean(np.stack(logVar_batch, axis=0), 0) < 0.1).sum()
        print(f"| Epoch: {epoch+1:<4}/{config['train']['num_epochs']:<4} | Train Loss: {losses[-1]:8.6f} | Test Loss: {test_losses[-1]:8.6f} | Best: {best_flag:<1} | Patience: {early_stop_counter:<3}/{config['train']['patience']} | Mode Collapsed: {mode_collapse:<3}/{config['latent_params']['latent_dim']:<3} | LR: {lr_now[0]:.6f} | Beta: {beta:.4f} |")
    end_time = time.time()
    print('Time taken for training: ', end_time - start_time)
    print('Time taken per epoch: ', (end_time - start_time) / num_epochs)
    
    return model, {"train_losses": losses, "test_losses": test_losses}


def bvae_encode(model, data, device):
    """
    Encode the data using the trained model.
    
    Args:
        model: The trained model.
        data: The input data to encode.
        device: The device to run the model on (CPU or GPU).
    
    Returns:
        Encoded data.
    """
    model.eval()
    with torch.no_grad():
        if not data.is_cuda:
            data = data.to(device, non_blocking=True)
        _, mean, _ = model(data)
    return mean

def bvae_decode(model, z, device):
    """
    Decode the latent representation using the trained model.
    
    Args:
        model: The trained model.
        z: The latent representation to decode.
        device: The device to run the model on (CPU or GPU).
    
    Returns:
        Decoded data.
    """
    model.eval()
    with torch.no_grad():
        if not z.is_cuda:
            z = z.to(device, non_blocking=True)
        reconstruction = model.decoder(z)
    return reconstruction



def bvae_batch_encode(model, data_path, latent_path, device, config, batch_size=1000):
    """
    Encode original data in batches for the latent space. 
    
    All coefficients saved to latent_path.
    Args:
        model: The trained model.
        data_path: Path to the input data in .h5 format.
        latent_path: Path to save the encoded latent space in h5 format.
        device: The device to run the model on (CPU or GPU).
    """

    import h5py
    import sys
    import pickle

    # get mean and std from latent_path replace 'coeff.h5' with 'scaler.pkl'
    latent_scaler_path = latent_path.replace('coeff.h5', 'scaler.pkl')
    if os.path.exists(latent_scaler_path):
        with open(latent_scaler_path, 'rb') as f:
            mean, std = pickle.load(f)
        print(f"Loaded mean and std from {latent_scaler_path}")

    

    
    # Load the data
    with h5py.File(data_path, 'r') as f: 
        num_samples = f['UV'].shape[0]
        data_shape = f['UV'].shape 
        num_batches = math.ceil(num_samples / batch_size)

        l_config = bvae_latent_config(config, data_shape)
        with open(latent_path.replace('.h5', '_config.pkl'), 'wb') as g:
            pickle.dump(l_config, g)
        print(f"Latent configuration saved to {latent_path.replace('.h5', '_config.pkl')}")

        data_mean = f['mean'][:]

        with h5py.File(latent_path, 'w') as l:
            if 'dofs' in l.keys():
                del l['dofs']
            l.create_dataset('dofs', (num_samples, config['latent_params']['latent_dim']), dtype='float32')

            for id in range(num_batches):
                snap_start = id * batch_size
                snap_end = (id + 1) * batch_size
                if snap_end >= num_samples:
                    snap_end = num_samples
                batch_size = snap_end - snap_start

                print(f"Processing batch {id + 1}/{num_batches} ({snap_start}:{snap_end})")
                sys.stdout.flush()

                print(f"Snapshots: {snap_start} to {snap_end}, batch size: {batch_size}")
        
                data = f['UV'][snap_start:snap_end, ...] - data_mean[np.newaxis, ...]

                data = normalize_data(data, mean, std)

                data = torch.tensor(data.transpose(0,3,1,2), dtype=torch.float32).to(device)
                
                model.eval()
                with torch.no_grad():
                    encoded_data = bvae_encode(model, data, device)
                    l['dofs'][snap_start:snap_end, :] = encoded_data.cpu().numpy()
    
    print(f"Latent space saved to {latent_path}")
    


class bvae_latent_config:
    """
    Save the latent configuration for the model.
    
    Args:
        model: The trained model.
        latent_dim: The dimension of the latent space.
    
    Returns:
        A dictionary containing the latent configuration.
    """
    def __init__(self, config, data_shape):
        self.latent_dim = config['latent_params']['latent_dim']
        self.num_modes = config['latent_params']['latent_dim']
        self.beta = config['latent_params']['beta']
        self.nx = data_shape[1]
        self.ny = data_shape[2]
        self.nx_t = self.nx
        self.ny_t = self.ny
        self.num_vars = data_shape[3]
        self.num_snaps = data_shape[0]

def bvae_batch_decode(model, dofs, rec_path, data_path, latent_path, device, batch_size=1000):
    """
    Decode the latent space in batches and save the reconstructed data.
    
    Args:
        model: The trained model.
        dofs: Path to the latent space data or a tensor of latent coefficients.
        rec_path: Path to save the reconstructed data.
        latent_path: Path to the latent space file.
        device: The device to run the model on (CPU or GPU).
        batch_size: Number of samples to process in each batch.
    """
    import h5py
    import numpy as np
    import sys

    # get mean and std from latent_path replace 'coeff.h5' with 'scaler.pkl'
    latent_scaler_path = latent_path.replace('coeff.h5', 'scaler.pkl')
    if os.path.exists(latent_scaler_path):
        with open(latent_scaler_path, 'rb') as f:
            mean, std = pickle.load(f)
        print(f"Loaded mean and std from {latent_scaler_path}")

    if isinstance(dofs, str):
        dof_path = dofs
        with h5py.File(dof_path, 'r') as f:
            dofs = f['dofs'][:]

    num_snaps = dofs.shape[0]
    num_batches = num_snaps // batch_size
    if num_snaps % batch_size != 0:
        num_batches += 1

    with h5py.File(data_path, 'r') as f:
        nx_t = f['UV'].shape[1]
        ny_t = f['UV'].shape[2]

    with h5py.File(rec_path, 'w') as rec_file:
        if 'Q_rec' in rec_file.keys():
            del rec_file['Q_rec']
        rec_file.create_dataset('Q_rec', (num_snaps, nx_t, ny_t, 2), dtype='float32')

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_snaps)
            sys.stdout.write(f"Reconstructing batch {i + 1}/{num_batches} ({start}:{end})")
            sys.stdout.flush()

            time_start = time.time()
            coeffs = torch.tensor(dofs[start:end, :], dtype=torch.float32).to(device)
            rec_data = bvae_decode(model, coeffs, device)
            rec_data = rec_data.cpu().numpy()
            rec_data = rec_data.transpose(0, 2, 3, 1)  # Convert to [samples, height, width, vars]
            rec_file['Q_rec'][start:end, :] = denormalize_data(rec_data, mean, std)

            time_end = time.time()
            sys.stdout.write(f"Batch {i + 1} processed in {time_end - time_start:.2f} seconds")
            sys.stdout.write('\n')
            sys.stdout.flush()



def bvae_mode_order(model, data_path, latent_path, config, device):
    import h5py
    with h5py.File(latent_path, 'r') as f:
        num_snaps = f['dofs'].shape[0]
        if num_snaps > 1000:
            num_snaps = 1000
        dofs = f['dofs'][:num_snaps, :]
        latent_dim = f['dofs'].shape[1]
    with h5py.File(data_path, 'r') as f:
        mean = f['mean'][:]
        Q = f['UV'][:num_snaps] - mean[np.newaxis, ...]
    
    with open(latent_path.replace('coeff.h5', 'scaler.pkl'), 'rb') as f:
        scaler_mean, scaler_std = pickle.load(f)
    
    m = np.zeros(latent_dim, dtype=int)
    n = np.arange(latent_dim)
    Ecum = []
    partialModes = np.zeros_like(dofs, dtype=np.float32)

    for i in range(latent_dim):
        Eks = []
        for j in n:  # for mode in remaining modes
            start = time.time()
            print(m[:i], j, end="")
            partialModes *= 0
            partialModes[:, m[:i]] = dofs[:, m[:i]]
            partialModes[:, j] = dofs[:, j]
            Q_pred = model.decoder(torch.tensor(partialModes, dtype=torch.float32).to(device))
            Q_pred = Q_pred.cpu().detach().numpy().transpose(0,2,3,1)
            Q_pred = denormalize_data(Q_pred, scaler_mean, scaler_std)
            Eks.append(get_Ek(Q, Q_pred))
            elapsed = time.time() - start
            print(f' : Ek={Eks[-1]:.4f}, elapsed: {elapsed:.2f}s')
        Eks = np.array(Eks).squeeze()
        ind = n[np.argmax(Eks)]
        m[i] = ind
        n = np.delete(n, np.argmax(Eks))
        Ecum.append(np.max(Eks))
        print('Adding: ', ind, ', Ek: ', np.max(Eks))
        print('#'*30)
    Ecum = np.array(Ecum)
    print(f"Rank finished, the rank is {m}")
    print(f"Cumulative Ek is {Ecum}")

    return np.array(m), Ecum

def get_Ek(original, rec):
    
    """
    Calculate energy percentage reconstructed
    
    Args:   
            original : (NumpyArray) The ground truth 

            rec      : (NumpyArray) The reconstruction from decoder

    Returns:  

            The energy percentage for construction. Note that it is the Ek/100 !!
    """

    import numpy as np 

    TKE_real = original[..., 0] ** 2 + original[..., 1] ** 2

    u_rec = rec[..., 0]
    v_rec = rec[..., 1]

    return 1 - np.sum((original[..., 0] - u_rec) ** 2 + (original[..., 1] - v_rec) ** 2) / np.sum(TKE_real)


## ====================================== Normalization ==========================================
def normalize_data(data, mean, std):
    return (data - mean) / std
def denormalize_data(data, mean, std):
    return (data * std) + mean
