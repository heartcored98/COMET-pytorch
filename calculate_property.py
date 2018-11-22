import multiprocessing as mp
import os
from os.path import isfile, join
import time
import argparse
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcPBF
import pandas as df

def calculate_property(smi):
    m = Chem.MolFromSmiles(smi)
    logP = MolLogP(m)
    mr = MolMR(m)
    tpsa = CalcTPSA(m)
    pbf = CalcPBF(m)



def process_file(filename):
    with open(filename) as file:
        list_row = file.readlines()[1:]

    temp_row = list()
    for row in tqdm(list_row):
        smi = row.split(' ')[0].strip()
        m = Chem.MolFromSmiles(smi)
        logP = MolLogP(m)
        mr = MolMR(m)
        tpsa = CalcTPSA(m)
        # pbf = CalcPBF(m)
        temp_row.append((smi, logP, mr, tpsa))


    return temp_row


def log_result(result):
    pass



def process_dataset(chunk_size=10000, #500000,
                    raw_dir_path='./dataset/raw_zinc_smiles',
                    output_dir_path='./dataset/processed_zinc_smiles',
                    start_offset=0, end_offset=-1, test_size=0.2):
    ts = time.time()

    # Start Downloading Process
    list_file = [f for f in os.listdir(raw_dir_path) if isfile(join(raw_dir_path, f))]
    # pool = mp.Pool(processes=10)

    cnt_train_mol = 0
    cnt_train_chunk = 0
    cnt_val_mol = 0
    cnt_val_chunk = 0
    train_row_buffer = list()
    val_row_buffer = list()
    for idx, filename in enumerate(list_file[start_offset:end_offset][:10]):
        print("Processing {}-th files".format(idx))
        data = process_file(join(raw_dir_path, filename))
        train_data, val_data = train_test_split(data,  test_size=test_size, random_state=111)
        train_row_buffer += train_data
        val_row_buffer += val_data
        cnt_train_mol += len(train_data)
        cnt_val_mol += len(val_data)


        if len(train_row_buffer) > chunk_size:
            output_train = train_row_buffer[:chunk_size]
            df_train = df.DataFrame.from_records(output_train, columns=('smile', 'logP', 'mr', 'tpsa'))
            df_train.to_csv(path_or_buf=join(output_dir_path, 'train/train{:06d}.csv'.format(cnt_train_chunk)), float_format='%g', index=False)
            train_row_buffer = train_row_buffer[chunk_size:]
            cnt_train_chunk += 1
            print("save train")
        if len(val_row_buffer) > chunk_size:
            output_val = val_row_buffer[:chunk_size]
            df_val = df.DataFrame.from_records(output_val, columns=('smile', 'logP', 'mr', 'tpsa'))
            df_val.to_csv(path_or_buf=join(output_dir_path, 'val/val{:06d}.csv'.format(cnt_val_chunk)), float_format='%g', index=False)
            val_row_buffer = val_row_buffer[chunk_size:]
            cnt_val_chunk += 1
            print("save val")


    df_train = df.DataFrame.from_records(train_row_buffer, columns=('smile', 'logP', 'mr', 'tpsa'))
    df_train.to_csv(path_or_buf=join(output_dir_path, 'train/train{:06d}.csv'.format(cnt_train_chunk)),
                        float_format='%g', index=False)
    df_val = df.DataFrame.from_records(val_row_buffer, columns=('smile', 'logP', 'mr', 'tpsa'))
    df_val.to_csv(path_or_buf=join(output_dir_path, 'val/val{:06d}.csv'.format(cnt_val_chunk)), float_format='%g',
                  index=False)

    # raise RuntimeError

        # time.sleep(0.05)
        # pool.apply_async(process_file, args = (join(raw_dir_path, filename),), callback=log_result)
    # pool.close()
    # pool.join()
    te = time.time()

    # Report Work
    print("=================================================================================")
    print("Processing Completed! Train : {} mols. Val : {} mols.  Took {:5.1f} sec".format(cnt_train_mol, cnt_val_mol, te-ts))
    print("=================================================================================")


def count_mol():
    mypath = './dataset/raw_zinc_smiles'
    onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
    print("=================================================================================")
    print("Counting Started!")
    print("=================================================================================")

    cnt_mol = 0
    for filename in tqdm(onlyfiles):
        cnt_mol += countline(join(mypath, filename))

    print("=================================================================================")
    print("Counting Completed!   {} Molecular retrieved from {} files.".format(cnt_mol, len(onlyfiles)))
    print("=================================================================================")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add logP, TPSA, MR, PBF value on .smi files')
    parser.add_argument("-n", "--num_worker", help="number of co-working process", type=int, default=8)
    parser.add_argument("-s", "--start_offset", help="starting from i-th file in directory", type=int, default=0)
    parser.add_argument("-e", "--end_offset", help="end processing at i-th file in directory", type=int, default=-1)
    parser.add_argument("-d", "--raw_dir_path", help="directory where dataset stored", type=str, default='./dataset/raw_zinc_smiles')
    parser.add_argument("-o", "--output_dir_path", help="directory where processed data saved", type=str, default='./dataset/processed_zinc_smiles')
    parser.add_argument("-r", "--test_size", help="portion of validation_set", type=float, default=0.2)

    args = parser.parse_args()

    # print(args)
    process_dataset(raw_dir_path=args.raw_dir_path, output_dir_path=args.output_dir_path, start_offset=args.start_offset, end_offset=args.end_offset, test_size=args.test_size)