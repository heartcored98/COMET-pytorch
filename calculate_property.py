import multiprocessing as mp
import os
from os.path import isfile, join
import time
import argparse
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcPBF

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
    for row in list_row:
        smi = row.split(' ')[0].strip()
        m = Chem.MolFromSmiles(smi)
        logP = MolLogP(m)
        mr = MolMR(m)
        tpsa = CalcTPSA(m)
        # pbf = CalcPBF(m)
        print(smi, logP, mr, tpsa) #, pbf)
    raise RuntimeError

def log_result(result):
    pass



def process_dataset(dir_path='./dataset/zinc_smiles', start_offset=0, end_offset=-1):
    ts = time.time()

    # Start Downloading Process
    list_file = [f for f in os.listdir(dir_path) if isfile(join(dir_path, f))]
    pool = mp.Pool(processes=10)
    for filename in list_file[start_offset:end_offset]:
        time.sleep(0.05)
        pool.apply_async(process_file, args = (join(dir_path, filename),), callback=log_result)
    pool.close()
    pool.join()
    te = time.time()

    # Report Work
    print("=================================================================================")
    print("Downloading Completed! {} Molecular retrieved from   {}/{} files. Took {:5.1f} sec".format(cnt_mol, total_file-cnt_fail, total_file, te-ts ))
    print("=================================================================================")


def count_mol():
    mypath = './dataset/zinc_smiles'
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
    parser.add_argument("-e", "--end_offset", help="end processing at i-th file in directory", type=int, default=1) #-1)
    parser.add_argument("-d", "--dir_path", help="directory where dataset stored", type=str, default='./dataset/zinc_smiles')
    args = parser.parse_args()

    print(args)
    process_dataset(dir_path=args.dir_path, start_offset=args.start_offset, end_offset=args.end_offset)