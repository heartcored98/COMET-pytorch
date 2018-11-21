import os
from os.path import isfile, join
import urllib.request
import multiprocessing as mp
import time
from tqdm import tqdm

# ==== Global Variables ==== #
mypath = './dataset/raw_zinc_smiles'
onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]

cnt_mol = 0
cnt_file = 0
cnt_fail = 0
total_file = 0
ts = time.time()

# ==== Downloading Function ==== #
def download(url, onlyfiles):
    name = url.split('/')[-1].strip() # filename would be AAAC.smi, AEBC.smi ...
    if name not in onlyfiles: # download file only when the file does not exist.
#         os.system('wget -q -P {} {}'.format('./dataset/raw_zinc_smiles', url))
        urllib.request.urlretrieve(url, './dataset/raw_zinc_smiles/{}'.format(name))
        return name, True
    return name, False

def countline(filename):
    try:
        with open(filename) as file:
            return len(file.readlines())
    except:
        return 0

def log_result(output):
    name = output[0]
    flag = output[1]
    cnt = countline('./dataset/raw_zinc_smiles/{}'.format(name))

    global cnt_mol, cnt_file
    cnt_file += 1
    cnt_mol += cnt
    took = time.time() - ts
    if cnt > 0:
        if flag:
            print("Downloaded {:8}. included {:9} mol. {:4}/{:4}. elapsed {:5.1f} sec".format(name, cnt, cnt_file, total_file, took))
        else:
            print("Already Ex {:8}. included {:9} mol. {:4}/{:4}. elapsed {:5.1f} sec".format(name, cnt, cnt_file, total_file, took))

    else:
        global cnt_fail
        cnt_fail += 1
        print("Failed     {:8}. included {:9} mol. {:4}/{:4}. elapsed {:5.1f} sec".format(name, cnt, cnt_file, total_file, took))


def save_smi(start_offset=0, filename='ZINC-downloader-2D-smi.uri'):
    ts = time.time()
    global cnt_file, total_file
    cnt_file += start_offset

    with open(filename) as file:
        list_url = file.readlines()
        total_file = len(list_url)

    # Reset dataset folder
    # os.system('rm -rf ./dataset/raw_zinc_smiles')
    # os.system('mkdir ./dataset/raw_zinc_smiles')

    # Start Downloading Process
    pool = mp.Pool(processes=10)
    for url in list_url[start_offset:]:
        time.sleep(0.05)
        pool.apply_async(download, args = (url, onlyfiles), callback=log_result)
    pool.close()
    pool.join()
    te = time.time()

    # Report Work
    print("=================================================================================")
    print("Downloading Completed! {} Molecular retrieved from   {}/{} files. Took {:5.1f} sec".format(cnt_mol, total_file-cnt_fail, total_file, te-ts ))
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
    # TODO : add argparse option - num_process, start_offset, only_counting_option, filename, download directory
    save_smi(start_offset=0)
    count_mol()