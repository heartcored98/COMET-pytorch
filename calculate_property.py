import multiprocessing as mp
import os
from os.path import isfile, join
import time
import argparse
from random import sample
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.rdMolDescriptors import CalcTPSA #,CalcPBF
import pandas as df
import re
import numpy as np
from numpy.random import choice
import gc


from dataloader import *

def extract_num(list_file):
    list_num = list()
    if len(list_file) == 0:
        return 0
    for file in list_file:
        list_num.append(int([float(s) for s in re.findall(r'-?\d+\.?\d*', file)][0]))
    return max(list_num)+1


def get_last_num(output_dir_path='./dataset/processed_zinc_smiles'):
    train_ouput_dir_path = join(output_dir_path, 'train')
    val_output_dir_path = join(output_dir_path, 'val')
    train_list_file = [f for f in os.listdir(train_ouput_dir_path) if isfile(join(train_ouput_dir_path, f))]
    val_list_file = [f for f in os.listdir(val_output_dir_path) if isfile(join(val_output_dir_path, f))]
    last_train_num = extract_num(train_list_file)
    last_val_num = extract_num(val_list_file)
    return last_train_num, last_val_num


def get_mol_importance(row):
    """Return Average Sampling Rate of Molecule based on symbol occurence"""
    try:
        smi = row.split(' ')[0].strip()
        word = "".join(re.findall("[a-zA-Z]+", smi)).lower()
        word = set(word)
        word.discard('c')
        word.discard('n')
        word.discard('o')
        if len(word) > 0:
            return (smi, len(word))
        return (smi, 1e-9)
    except:
        return (None, None)


def process_smile(row):
    """Return molecular properties """
    try:
        smi = row[0]
        m = Chem.MolFromSmiles(smi)
        logP = MolLogP(m)
        mr = MolMR(m)
        tpsa = CalcTPSA(m)
        n_atom = m.GetNumAtoms()
        del m
        return smi, logP, mr, tpsa, n_atom
    except:
        return None, None, None, None, None


def process_dataset(chunk_size,
                    temp_size,
                    num_worker,
                    flag_continue,
                    sampling_rate,
                    test_size,
                    raw_dir_path,
                    output_dir_path,
                    start_offset,
                    end_offset):
    ts = time.time()

    list_file = [f for f in os.listdir(raw_dir_path) if isfile(join(raw_dir_path, f))]
    cnt_train_mol = 0
    cnt_val_mol = 0

    # Get Starting Point if processing is continued
    if flag_continue:
        cnt_train_chunk, cnt_val_chunk = get_last_num(output_dir_path)
    else:
        cnt_train_chunk, cnt_val_chunk = 0, 0

    # Initialize List
    train_row_buffer = list()
    val_row_buffer = list()
    label_columns = ('smile', 'logP', 'mr', 'tpsa', 'length')
    target_list_file = list_file[start_offset:end_offset]
    threshold = int(1 / sampling_rate) * 100

    last_train_ck = 0
    last_val_ck = 0

    # Iterating raw zinc dataset and parse it.
    for idx, filename in enumerate(target_list_file):
        m_ts = time.time()
        with open(join(raw_dir_path, filename)) as file:
            list_row = file.readlines()[1:]
        if len(list_row) > threshold:

            # Calculate individual sampling rate of molecule
            with mp.Pool(processes=num_worker) as pool:
                row_prob = np.array(pool.map(get_mol_importance, list_row))
            samplings = np.nan_to_num(np.array([row[1] for row in row_prob], dtype=np.float), 0)
            samplings /= np.sum(samplings)

            # Sample molecules based on each importance
            sampled_idx = choice(range(len(row_prob)), max(int(len(row_prob)*sampling_rate), 1), replace=False, p=samplings)
            sampled_rows = row_prob[sampled_idx]
            with mp.Pool(processes=num_worker) as pool:
                data = pool.map(process_smile, sampled_rows)

            # Split dataset into training and validation set
            train_data, val_data = train_test_split(data,  test_size=test_size)
            train_row_buffer += train_data
            val_row_buffer += val_data
            cnt_train_mol += len(train_data)
            cnt_val_mol += len(val_data)

            if len(train_row_buffer) < chunk_size and len(train_row_buffer) // temp_size > last_train_ck:
                df_train = df.DataFrame.from_records(train_row_buffer, columns=label_columns)
                df_train = df_train.dropna()
                df_train.sort_values(by=['length'], ascending=False, inplace=True)
                df_train.to_csv(path_or_buf=join(output_dir_path, 'train/train{:06d}.csv'.format(cnt_train_chunk)),
                                float_format='%g', index=False)
                last_train_ck = len(train_row_buffer) // temp_size
                print("Train Checkpoint Saved with train{:06d}.csv CK:{}".format(cnt_train_chunk, last_train_ck))


            if len(val_row_buffer) < chunk_size and len(val_row_buffer) // temp_size > last_val_ck:
                df_val = df.DataFrame.from_records(val_row_buffer, columns=label_columns)
                df_val = df_val.dropna()
                df_val.sort_values(by=['length'], ascending=False, inplace=True)
                df_val.to_csv(path_or_buf=join(output_dir_path, 'val/val{:06d}.csv'.format(cnt_val_chunk)),
                              float_format='%g', index=False)
                last_val_ck = len(val_row_buffer) // temp_size
                print("Valid Checkpoint Saved with   val{:06d}.csv CK:{}".format(cnt_val_chunk, last_val_ck))

            while len(train_row_buffer) > chunk_size:
                if len(train_row_buffer) >= chunk_size:
                    output_train = train_row_buffer[:chunk_size]
                    df_train = df.DataFrame.from_records(output_train, columns=label_columns)
                    df_train = df_train.dropna()
                    df_train.sort_values(by=['length'], ascending=False, inplace=True)
                    df_train.to_csv(path_or_buf=join(output_dir_path, 'train/train{:06d}.csv'.format(cnt_train_chunk)), float_format='%g', index=False)
                    train_row_buffer = train_row_buffer[chunk_size:]
                    cnt_train_chunk += 1

                if len(val_row_buffer) >= chunk_size:
                    output_val = val_row_buffer[:chunk_size]
                    df_val = df.DataFrame.from_records(output_val, columns=label_columns)
                    df_val = df_val.dropna()
                    df_val.sort_values(by=['length'], ascending=False, inplace=True)
                    df_val.to_csv(path_or_buf=join(output_dir_path, 'val/val{:06d}.csv'.format(cnt_val_chunk)), float_format='%g', index=False)
                    val_row_buffer = val_row_buffer[chunk_size:]
                    cnt_val_chunk += 1
        else:
            data = []

        m_te = time.time()
        gc.collect()
        print("Processed {} {:4}/{:4} files of {:8} mols took {:5.1f} sec. {:7.1f} mol/sec. Train: {:4} Val: {:4} Elap: {:6.1f} sec".format(
                filename, idx + 1, len(target_list_file), len(data), m_te - m_ts, len(data) / (m_te - m_ts), cnt_train_chunk+1,
                cnt_val_chunk+1, time.time() - ts))

    # Convert dataset into Dataframe and save it into csv format
    df_train = df.DataFrame.from_records(train_row_buffer, columns=label_columns)
    df_train = df_train.dropna()
    df_train.sort_values(by=['length'], ascending=False, inplace=True)
    df_train.to_csv(path_or_buf=join(output_dir_path, 'train/train{:06d}.csv'.format(cnt_train_chunk)),
                        float_format='%g', index=False)
    df_val = df.DataFrame.from_records(val_row_buffer, columns=label_columns)
    df_val = df_val.dropna()
    df_val.sort_values(by=['length'], ascending=False, inplace=True)
    df_val.to_csv(path_or_buf=join(output_dir_path, 'val/val{:06d}.csv'.format(cnt_val_chunk)), float_format='%g',
                  index=False)

    te = time.time()

    # Report Work
    print("======================================================================================")
    print("Processing Completed! Train : {} mols. Val : {} mols.  Took {:5.1f} sec. {:5.1f} mol/sec".format(cnt_train_mol, cnt_val_mol, te-ts, (cnt_train_mol+cnt_val_mol)/(te-ts)))
    print("======================================================================================")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add logP, TPSA, MR, PBF value on .smi files')
    # S : 0.0047049 -> 2M / 0.5M
    # XS : 0.0047049 * 0.3 -> 0.6M / 0.15M
    # XXS : 0.00047049 -> 0.2M / 0.05M
    parser.add_argument("-q", "--sampling_rate", help="sampling rate", type=float, default=0.0047049)

    parser.add_argument("-c", "--chunk_size", help="number of rows in one chunk ", type=int, default=25000000)
    parser.add_argument("-ts", "--temp_size", help="number of rows in one chunk ", type=int, default=50000000)
    parser.add_argument("-n", "--num_worker", help="number of co-working process", type=int, default=16)
    parser.add_argument("-r", "--test_size", help="portion of validation_set", type=float, default=0.2)
    parser.add_argument("-t", "--flag_continue", help="whether continue writing file", type=bool, default=True)

    parser.add_argument("-s", "--start_offset", help="starting from i-th file in directory", type=int, default=0)
    parser.add_argument("-e", "--end_offset", help="end processing at i-th file in directory", type=int, default=-1)
    parser.add_argument("-d", "--raw_dir_path", help="directory where dataset stored", type=str, default='./raw_zinc_smiles')
    parser.add_argument("-o", "--output_dir_path", help="directory where processed data saved", type=str, default='./dataset/bal_s')

    args = parser.parse_args()


    process_dataset(chunk_size=args.chunk_size,
                    temp_size=args.temp_size,
                    num_worker=args.num_worker,
                    sampling_rate=args.sampling_rate,
                    flag_continue=args.flag_continue,
                    raw_dir_path=args.raw_dir_path,
                    output_dir_path=args.output_dir_path,
                    start_offset=args.start_offset,
                    end_offset=args.end_offset,
                    test_size=args.test_size)


