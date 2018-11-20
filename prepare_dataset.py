import os
from os.path import isfile, join
import multiprocessing as mp
import time

# ==== Global Variables ==== #
mypath = './dataset/zinc_smiles'
onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]

cnt_mol = 0
cnt_file = 0
cnt_fail = 0
total_file = 0
ts = time.time()

# ==== Downloading Function ==== #
def download(url):
    name = url.split('/')[-1].strip() # filename would be AAAC.smi, AEBC.smi ...
    if name not in onlyfiles: # download file only when the file does not exist.
        os.system('wget -q -P {} {}'.format('./dataset/zinc_smiles', url))
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
    cnt = countline('./dataset/zinc_smiles/{}'.format(name))

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


def save_smi(filename='ZINC-downloader-2D-smi.uri'):
    ts = time.time()

    with open(filename) as file:
        list_url = file.readlines()

    global total_file
    total_file = len(list_url)

    pool = mp.Pool(processes=40)
    # os.system('rm -rf ./dataset/zinc_smiles')
    # os.system('mkdir ./dataset/zinc_smiles')
    for url in list_url:
        pool.apply_async(download, args = (url,), callback=log_result)
    pool.close()
    pool.join()
    te = time.time()

    print("=================================================================================")
    print("Download Completed!   {} Molecular retrieved from   {}/{} files. Took {:5.1f} sec".format(cnt_mol, total_file-cnt_fail, total_file, te-ts ))
    print("=================================================================================")


if __name__ == '__main__':

    save_smi()
