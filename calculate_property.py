import multiprocessing as mp
import os
from os.path import isfile, join
import time
import argparse
from random import sample
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcPBF
import pandas as df
import re

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

def process_smile(row):
    try:
        smi = row.split(' ')[0].strip()
        m = Chem.MolFromSmiles(smi)
        return (smi, MolLogP(m), MolMR(m), CalcTPSA(m), m.GetNumAtoms())
    except:
        return (None, None, None, None, None)

def process_dataset(chunk_size,
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
    if flag_continue:
        cnt_train_chunk, cnt_val_chunk = get_last_num(output_dir_path)
    else:
        cnt_train_chunk, cnt_val_chunk = 0, 0
    train_row_buffer = list()
    val_row_buffer = list()
    label_columns = ('smile', 'logP', 'mr', 'tpsa', 'length')
    target_list_file = list_file[start_offset:end_offset]


    for idx, filename in enumerate(target_list_file):
        m_ts = time.time()
        with open(join(raw_dir_path, filename)) as file:
            list_row = file.readlines()[1:]
            sampled_list_row = sample(list_row, int(len(list_row)*sampling_rate))
            with mp.Pool(processes=num_worker) as pool:
                data = pool.map(process_smile, sampled_list_row)
        train_data, val_data = train_test_split(data,  test_size=test_size, random_state=111)
        train_row_buffer += train_data
        val_row_buffer += val_data
        cnt_train_mol += len(train_data)
        cnt_val_mol += len(val_data)

        while len(train_row_buffer) > chunk_size:
            if len(train_row_buffer) >= chunk_size:
                output_train = train_row_buffer[:chunk_size]
                df_train = df.DataFrame.from_records(output_train, columns=label_columns)
                df_train = df_train.dropna()
                df_train.to_csv(path_or_buf=join(output_dir_path, 'train/train{:06d}.csv'.format(cnt_train_chunk)), float_format='%g', index=False)
                train_row_buffer = train_row_buffer[chunk_size:]
                cnt_train_chunk += 1

            if len(val_row_buffer) >= chunk_size:
                output_val = val_row_buffer[:chunk_size]
                df_val = df.DataFrame.from_records(output_val, columns=label_columns)
                df_val = df_val.dropna()
                df_val.to_csv(path_or_buf=join(output_dir_path, 'val/val{:06d}.csv'.format(cnt_val_chunk)), float_format='%g', index=False)
                val_row_buffer = val_row_buffer[chunk_size:]
                cnt_val_chunk += 1

        m_te = time.time()
        print("Processed {:4}/{:4} files of {:8} mols took {:5.1f} sec. {:7.1f} mol/sec. Train: {:4} Val: {:4} Elap: {:6.1f} sec".format(
                idx + 1, len(target_list_file), len(data), m_te - m_ts, len(data) / (m_te - m_ts), cnt_train_chunk+1,
                cnt_val_chunk+1, time.time() - ts))

    df_train = df.DataFrame.from_records(train_row_buffer, columns=label_columns)
    df_train = df_train.dropna()
    df_train.to_csv(path_or_buf=join(output_dir_path, 'train/train{:06d}.csv'.format(cnt_train_chunk)),
                        float_format='%g', index=False)
    df_val = df.DataFrame.from_records(val_row_buffer, columns=label_columns)
    df_val = df_val.dropna()
    df_val.to_csv(path_or_buf=join(output_dir_path, 'val/val{:06d}.csv'.format(cnt_val_chunk)), float_format='%g',
                  index=False)

    te = time.time()

    # Report Work
    print("======================================================================================")
    print("Processing Completed! Train : {} mols. Val : {} mols.  Took {:5.1f} sec. {:5.1f} mol/sec".format(cnt_train_mol, cnt_val_mol, te-ts, (cnt_train_mol+cnt_val_mol)/(te-ts)))
    print("======================================================================================")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add logP, TPSA, MR, PBF value on .smi files')
    parser.add_argument("-c", "--chunk_size", help="number of rows in one chunk ", type=int, default=25000000)
    parser.add_argument("-n", "--num_worker", help="number of co-working process", type=int, default=16)
    parser.add_argument("-q", "--sampling_rate", help="number of co-working process", type=float, default=0.001)
    parser.add_argument("-s", "--start_offset", help="starting from i-th file in directory", type=int, default=0)
    parser.add_argument("-e", "--end_offset", help="end processing at i-th file in directory", type=int, default=-1)
    parser.add_argument("-d", "--raw_dir_path", help="directory where dataset stored", type=str, default='./dataset/raw_zinc_smiles')
    parser.add_argument("-o", "--output_dir_path", help="directory where processed data saved", type=str, default='./dataset/processed_zinc_smiles/data_xs')
    parser.add_argument("-r", "--test_size", help="portion of validation_set", type=float, default=0.2)
    parser.add_argument("-t", "--flag_continue", help="whether continue writing file", type=bool, default=False)

    args = parser.parse_args()


    process_dataset(chunk_size=args.chunk_size,
                    num_worker=args.num_worker,
                    sampling_rate=args.sampling_rate,
                    flag_continue=args.flag_continue,
                    raw_dir_path=args.raw_dir_path,
                    output_dir_path=args.output_dir_path,
                    start_offset=args.start_offset,
                    end_offset=args.end_offset,
                    test_size=args.test_size)


