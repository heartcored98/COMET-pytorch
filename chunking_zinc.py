import os
from os.path import isfile, join
import time
import argparse


def process_dataset(chunk_size,
                    raw_dir_path,
                    start_offset,
                    end_offset):
    ts = time.time()
    list_file = [f for f in os.listdir(raw_dir_path) if isfile(join(raw_dir_path, f))]
    print(list_file)
    cnt_file = 0
    os.chdir(raw_dir_path)

    # Initialize List
    # list_file = list_file[start_offset:end_offset]

    # Iterating raw zinc dataset and parse it.
    for idx, filename in enumerate(list_file):
        with open(filename) as file:
            list_row = file.readlines()[1:]
        if len(list_row) > chunk_size:
            cnt_file += 1
            cmd = 'split -l {:d} -d -e {} {}_'.format(chunk_size, filename, filename)
            os.system(cmd)
            cmd = 'rm {}'.format(filename)
            os.system(cmd)
            print("Processed {} {:4}/{:4} files  Elap: {:6.1f} sec".format(
                    filename, idx + 1, len(list_file), time.time() - ts))

        else:
            print("Passed    {} {:4}/{:4} files  Elap: {:6.1f} sec".format(
                    filename, idx + 1, len(list_file), time.time() - ts))



    te = time.time()

    # Report Work
    print("======================================================================================")
    print("Processing Completed! Processed {} files  Took {:5.1f} sec. ".format(cnt_file, te-ts))
    print("======================================================================================")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add logP, TPSA, MR, PBF value on .smi files')

    parser.add_argument("-c", "--chunk_size", help="number of rows in one chunk ", type=int, default=650000)
    parser.add_argument("-s", "--start_offset", help="starting from i-th file in directory", type=int, default=0)
    parser.add_argument("-e", "--end_offset", help="end processing at i-th file in directory", type=int, default=-1)
    parser.add_argument("-d", "--raw_dir_path", help="directory where dataset stored", type=str, default='./raw_zinc_smiles')

    args = parser.parse_args()


    process_dataset(chunk_size=args.chunk_size,
                    raw_dir_path=args.raw_dir_path,
                    start_offset=args.start_offset,
                    end_offset=args.end_offset)


