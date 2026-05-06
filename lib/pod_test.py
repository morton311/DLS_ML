import numpy as np
import h5py
from lib.pod import pod_mode_find, pod_decomp

data_path, latent_path = 'data/ldc_30k_60ksnaps.h5', 'results/ldc_30k_60ksnaps/pod/latent_coeffs.h5'

with h5py.File(data_path, 'r') as f:
    mean = f['mean'][:]
    data = f['UV'][:2500:2] - mean[np.newaxis, ...] 

modes, eigVal, config = pod_mode_find(data)
print(f"Modes shape: {modes.shape}")
print(f"Eigenvalues shape: {eigVal.shape}")
print(f"Config: {config.__dict__}")
print("POD decomposition completed successfully.")

with h5py.File(latent_path, 'w') as f:
    f.create_dataset('eigVal', data=eigVal)
    f.create_dataset('modes', data=modes)