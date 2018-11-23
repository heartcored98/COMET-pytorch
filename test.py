from os.path import isfile, join
import time
import h5py
import numpy as np
import random
from tqdm import tqdm

from download_zinc import count_mol

def write_record(records,
                 overwrite = False,
                 chunk_size = True,
                 filename='processed_zinc.hdf5',
                 dir_path='./dataset/processed_zinc_smiles'):

    smi_records, property_records, length_records = records
    property_dim = property_records.shape[1]
    records_size = property_records.shape[0]
    assert smi_records.shape[0] == property_records.shape[0] and smi_records.shape[0] == length_records.shape[0], \
        "Records Dimension is different!"

    access = 'w' if overwrite else 'a'
    with h5py.File(join(dir_path, filename), access) as f:

        # Initializing Dataset
        list_dataset = list(f.keys())
        if len(list_dataset) == 0:
            chunk_size1 = chunk_size if chunk_size is True else (chunk_size,)
            chunk_size2 = chunk_size if chunk_size is True else (chunk_size, property_dim)

            smi = f.create_dataset('smile', (1,), maxshape=(None,), chunks=chunk_size1, dtype='S10')#,
                                   # compression="gzip", compression_opts=9)
            property = f.create_dataset('property', (1, property_dim), maxshape=(None, property_dim), chunks=chunk_size2, dtype='f')#,
                                        # compression="gzip", compression_opts=9)
            length = f.create_dataset('length', (1,), maxshape=(None,), chunks=chunk_size1, dtype='i4')#,
                                      # compression="gzip", compression_opts=9)
            smi.attrs['size'] = 0
            property.attrs['size'] = 0
            length.attrs['size'] = 0

        smi = f['smile']
        property = f['property']
        length = f['length']

        dataset_size = smi.attrs['size']
        assert smi.attrs['size'] == property.attrs['size'] and  smi.attrs['size'] == length.attrs['size'], \
            "Dataset Dimension is different!"

        smi.resize((dataset_size+records_size,))
        property.resize((dataset_size+records_size, property_dim))
        length.resize((dataset_size+records_size,))

        smi[dataset_size:dataset_size+records_size] = data_smile
        property[dataset_size:dataset_size+records_size] = data_property
        length[dataset_size:dataset_size+records_size] = data_length

        smi.attrs['size'] += records_size
        property.attrs['size'] += records_size
        length.attrs['size'] += records_size

def read_record(idx,
                filename='processed_zinc.hdf5',
                dir_path='./dataset/processed_zinc_smiles'):

    with h5py.File(join(dir_path, filename), 'r') as f:
        smi = f['smile']
        property = f['property']
        length = f['length']
        return smi[idx], property[idx][:], length[idx]


if __name__ == '__main__':
    datasize = 10000
    data_property = np.random.rand(datasize, 3)
    data_smile = list()
    for i in range(datasize):
        data_smile.append("asdfasdfas")

    data_smile = [n.encode("ascii", "ignore") for n in data_smile]
    data_smile = np.array(data_smile)

    data_length = list()
    for i in range(datasize):
        data_length.append(455)
    data_length = np.array(data_length)

    ts = time.time()
    wshot = 10
    for i in tqdm(range(wshot)):
        # pass
        write_record(records=(data_smile, data_property, data_length), overwrite=False) #, chunk_size=500)
    te = time.time()
    print("Took", te-ts, datasize*wshot / (te-ts), "record/sec")
    shot = 10000
    for i in tqdm(range(shot)):
        resp = read_record(i) #(idx=random.randint(0, datasize*wshot))
    te2 = time.time()
    print("Took", te2-te, shot / (te2-te), "record/sec")


