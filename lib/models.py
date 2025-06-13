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

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=time_lag)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
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
        hidden = torch.zeros(self.num_layer,
                                batch_size,
                                self.hidden_size).to(device)
                    
        cell  =  torch.zeros(self.num_layer,
                                batch_size,
                                self.hidden_size).to(device) 
                    
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

