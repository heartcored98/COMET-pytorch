
import h5py
import numpy as np

data_property = np.random.rand(3, 1000)

with h5py.File('processed_zinc.hdf5', 'w') as f:
    smi = f.create_dataset('smile', (1,1), maxshape=(1, None),  dtype='S10', compression="gzip", compression_opts=9)
    property = f.create_dataset('property', (3,1), maxshape=(3, None),  dtype='f16', compression="gzip", compression_opts=9)
    length = f.create_dataset('length', (1,1), maxshape=(1, None),  dtype='i4', compression="gzip", compression_opts=9)
    # d[:] = arr
