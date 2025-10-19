import torch
import torch.nn as nn
import math
from functools import partial
from datetime import datetime
import os
import numpy as np
import time
from tqdm import tqdm
import copy
import pickle


## ==================================== Positional Encoding ======================================
# Positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    



## ===================================== TimeSpace Embed =========================================

"""

Create a new embedding strategy for time and space embedding

@ yuningw

"""

class TimeSpaceEmbedding(nn.Module):
    
    """"

    A embedding module based on both time and space
    Args:

    d_input : The input size of timedelay embedding

    n_mode : The number of modes/dynamics in the time series 

    d_expand : The projection along the time

    d_model : The projection along the space 

    """

    def __init__(self, time_lag, input_dim,
                d_expand, d_model):

        super(TimeSpaceEmbedding, self).__init__()

        self.spac_proj      = nn.Linear(input_dim,d_model)

        self.time_proj      = nn.Conv1d(time_lag, d_expand,1)

        self.time_avgpool   = nn.AvgPool1d(2)
        self.time_maxpool   = nn.MaxPool1d(2)
        self.time_compress  = nn.Linear(d_model, d_model)
        self.act            = nn.Identity()

        nn.init.xavier_uniform_(self.spac_proj.weight)
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.xavier_uniform_(self.time_compress.weight)
    
    def forward(self, x):
        
        # Along space projection
        x       = self.spac_proj(x)
        
        # Along the time embedding 
        x       = self.time_proj(x)
        timeavg = self.time_avgpool(x)
        timemax = self.time_maxpool(x)
        tau     = torch.cat([timeavg, timemax],-1)
        out     = self.act(self.time_compress(tau))
        return out

## ====================================== Transformer ============================================
# Define the Transformer Encoder model
class TransformerEncoderModel(nn.Module):
    def __init__(self, time_lag, input_dim, d_model=256, nhead=4, num_layers=4, embed='lin'):
        super(TransformerEncoderModel, self).__init__()
        if embed == 'TS':
            self.positional_encoding = nn.Identity()
            self.input_projection = TimeSpaceEmbedding(time_lag, input_dim, d_expand=2 * time_lag, d_model=d_model)
        elif embed == 'lin':
            self.positional_encoding = PositionalEncoding(d_model, max_len=time_lag)
            self.input_projection = nn.Linear(input_dim, d_model)
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, input_dim)

        # Attention outputs storage
        self.encoder_attn_outputs = {}
        self.patch_attention()

    def patch_attention_layer(self, m):
        """Monkey-patch the attention layer to save attention weights."""
        forward_orig = m.forward

        def wrap(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return forward_orig(*args, **kwargs)

        m.forward = wrap

    def patch_attention(self):
        """Patch all attention layers in the encoder."""
        for i, layer in enumerate(self.encoder_layers):
            self.patch_attention_layer(layer.self_attn)
            layer.self_attn.register_forward_hook(partial(self.save_output_encoder, label=f's{i}'))

    def save_output_encoder(self, m, i, o, label='0'):
        """Save the attention weights from the encoder."""
        self.encoder_attn_outputs[label] = o[1].cpu().detach()

    def get_attn(self):
        """Retrieve the saved attention weights."""
        return self.encoder_attn_outputs.copy()

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)
            
        x = self.fc(x[:, -1, :])
        return x
    

## ====================================== LSTM Model ============================================
class LSTMModel(nn.Module):
    def __init__(self, time_lag, input_dim, hidden_dim=256, num_layers=2, batch_size = 256):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=time_lag)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        # Initialize hidden and cell states
        hidden, cell = self.init_hidden(x.shape[0], x.device)
        lstm_out, _ = self.lstm(x, (hidden.detach(), cell.detach()))
        out = self.fc(lstm_out[:, -1, :])
        return out
    
    def init_hidden(self,batch_size,device):
        hidden = torch.zeros(self.num_layers,
                                batch_size,
                                self.hidden_dim).to(device)
                    
        cell  =  torch.zeros(self.num_layers,
                                batch_size,
                                self.hidden_dim).to(device) 
                    
        return hidden, cell


## ====================================== Train Model ============================================

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, patience, device, model_dir, data_name):
    """
    Train the model with the given parameters.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train.
        patience (int): Early stopping patience.
        device (torch.device): Device to use for training (CPU or GPU).
        model_dir (str): Directory to save the model.
        data_name (str): Name of the dataset for saving the model.

    Returns:
        dict: A dictionary containing training and test losses.
    """
    best_test_loss = float('inf')
    early_stop_counter = 0
    losses = []
    test_losses = []

    # Generate a timestamp for saving the model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = os.path.join(model_dir, f'model_{timestamp}.pth')

    # Training loop
    start_time = time.time()
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        ## --------------------------------------- Train ---------------------------------------
        for inputs, targets in train_loader: 
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            total_loss = 0.0

            for n in range(targets.shape[1]):
                target = targets[:, n, :]  # shape: [B, input_dim]

                # Forward pass
                outputs = model(inputs)  # shape: [B, input_dim]
                loss = criterion(outputs, target)

                # Backward and optimization for current step only
                total_loss += loss
                loss.backward()

                # Prepare input for next step
                inputs = torch.cat((inputs[:, 1:, :], outputs.detach().unsqueeze(1)), dim=1)

            epoch_loss += total_loss.item() / targets.shape[1]
            optimizer.step()

        losses.append(epoch_loss / len(train_loader))

        ## --------------------------------------- Test ---------------------------------------
        # Evaluate the model on the test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                for n in range(targets.shape[1]):
                    target = targets[:, n, :]
                    outputs = model(inputs)
                    loss = criterion(outputs, target)
                    test_loss += loss.item() / targets.shape[1]
                    inputs = torch.cat((inputs[:, 1:, :], outputs.unsqueeze(1)), dim=1)

        test_losses.append(test_loss / len(test_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

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

                # Save the best model checkpoint
                checkpoint_path = os.path.join(model_dir, f'{data_name}_best_model.pth')
                torch.save(best_model, checkpoint_path)
                best_epoch = epoch + 1
                print(f'Best model saved at epoch {best_epoch} with test loss: {best_test_loss:.4f}')
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    model.load_state_dict(best_model)
                    print(f'Best model loaded from epoch {best_epoch}, with test loss: {best_test_loss:.4f}')
                    break

    end_time = time.time()
    print('Time taken for training: ', end_time - start_time)
    print('Time taken per epoch: ', (end_time - start_time) / num_epochs)

    # Save the final model after training
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")

    return {"train_losses": losses, "test_losses": test_losses}


## ====================================== Normalization ==========================================
def normalize_data(data, mean, std):
    return (data - mean) / std
def denormalize_data(data, mean, std):
    return (data * std) + mean

## ====================================== Model Prediction ==========================================
# long time history prediction
def predict(model, initial_input, time_lag, num_predictions, device):
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

    if initial_input.shape[0] != 1:
        initial_input = initial_input[np.newaxis, ...]
    if initial_input.shape[1] != time_lag:
        print(f"Initial input shape {initial_input.shape[1]} does not match time lag {time_lag}, input will be trucated.")
        initial_input = initial_input[:, :time_lag, :]
    current_input = torch.tensor(initial_input, dtype=torch.float32).to(device)

    model.eval()
    predictions = []

    start_time = time.time()
    with torch.no_grad():
        for _ in tqdm(range(num_predictions)):
            output = model(current_input)
            # print('Output shape: ', output[np.newaxis,:,:].shape)
            predictions.append(output.to('cpu').numpy())  # Ensure tensor is moved to CPU before converting to NumPy
            # Update the input for the next prediction
            current_input = torch.cat((current_input[:,1:, :], output[np.newaxis,:,:]), dim=1)
            # print('Current input shape: ', current_input.shape)

    end_time = time.time()
    print('Time taken for long-term prediction: ', end_time - start_time)
    print('Time taken per prediction: ', (end_time - start_time)/num_predictions)
    predictions = np.array(predictions)  # Convert list to NumPy array
    predictions = predictions.reshape(num_predictions, -1)  # Reshape to (num_predictions, input_dim)
    predictions = np.vstack([initial_input[0,:,:], predictions])  # Concatenate 
    return predictions

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