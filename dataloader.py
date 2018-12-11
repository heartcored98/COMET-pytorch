
from os.path import join
import multiprocessing as mp

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler
from torch._six import int_classes as _int_classes
from rdkit import Chem
from scipy.linalg import fractional_matrix_power
import numpy as np
import pandas as pd

from utils import get_dir_files


MASKING_RATE = 0
ERASE_RATE = 0
MAX_LEN = 50

LIST_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']

LIST_PROB = [2.3993241881917855e-05, 8.444776409026159e-09, 5.054594488705504e-08, 6.021403652679174e-08,
             4.649727426452611e-07, 3.068536033477748e-07, 0.03448275862068955e-09, 0.0005595207228372417,
             0.0006275408891429457, 9.129742249043719e-07, 1.9841406396876806e-06, 0.034482758620689655,
             0.034482758620689655, 0.034482758620689655, 0.034482758620689655, 0.034482758620689655,
             0.034482758620689655, 0.034482758620689655, 0.034482758620689655, 2.2842470449140126e-05,
             0.0012404878041197762, 0.034482758620689655, 0.034482758620689655, 0.034482758620689655,
             0.03200458534629023, 0.034482758620689655, 0.034482758620689655, 0.034482758620689655,
             0.034482758620689655, 0.034482758620689655, 0.034482758620689655, 0.034482758620689655,
             0.034482758620689655, 0.034482758620689655, 0.034482758620689655, 0.034482758620689655,
             0.034482758620689655, 0.034482758620689655, 0.034482758620689655, 0.034482758620689655,
             0.034482758620689655]

LOGP_MEAN, LOGP_STD = 3.0475299537604004, 1.4508318866361838
MR_MEAN, MR_STD = 1.983070758071883, 0.07702976853699765
TPSA_MEAN, TPSA_STD = 1.8082864863018322, 0.1832254436608209


def atom_feature(atom):
    return np.array(char_to_ix(atom.GetSymbol(), LIST_SYMBOLS) +
                    char_to_ix(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    char_to_ix(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    char_to_ix(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    char_to_ix(int(atom.GetIsAromatic()), [0, 1]))    # (40, 6, 5, 6, 2)


def char_to_ix(x, allowable_set):
    if x not in allowable_set:
        return [0] # Unknown Atom Token
    return [allowable_set.index(x)+1]


def normalize_adj(mx):
    """ Symmetry Normalization """
    rowsum = np.diag(np.array(mx.sum(1)))
    r_inv = fractional_matrix_power(rowsum, -0.5)
    r_inv[np.isinf(r_inv)] = 0.
    return r_inv.dot(mx).dot(r_inv)


def masking_feature(feature, num_masking, erase_rate, list_prob):
    """ Given feature, select 'num_masking' node feature and perturbate them.
    
        [5 features : Atom symbol, degree, num Hs, valence, isAromatic]  
        were masked with zero or changed with random one-hot encoding 
        or remained with origianl data(but still should be predicted).
        
        Masking process was conducted on each feature indiviually. 
        For example, if ERASE_RATE = 0.5, probability for all feature information with zero is 0.5^5 = 0.03125
        
        return original hode feature with their corresponding indices
    """
    ERASE_RATE = erase_rate
    
    masking_indices = np.random.choice(len(feature), num_masking, replace=False, p=list_prob / np.sum(list_prob))
    ground_truth = np.copy(feature[masking_indices, :])
    masked_feature = np.copy(feature)
    prob_masking = np.random.rand(len(masking_indices))
    for idx, i in enumerate(masking_indices):

        # Masking All Feature
        if prob_masking[idx] < ERASE_RATE:
            masked_feature[i, :] = 0

        # Otherwise, replace with random feature
        elif prob_masking[0] > 1- ((1-ERASE_RATE) * 0.5):
            masked_feature[i, 0] = np.random.randint(1, 41)
            masked_feature[i, 1] = np.random.randint(1, 7)
            masked_feature[i, 2] = np.random.randint(1, 6)
            masked_feature[i, 3] = np.random.randint(1, 7)
            masked_feature[i, 4] = np.random.randint(1, 3)

    return masked_feature, ground_truth, masking_indices


def mol2graph(smi):
    mol = Chem.MolFromSmiles(smi)

    X = np.zeros((MAX_LEN, 5), dtype=np.uint8)
    A = np.zeros((MAX_LEN, MAX_LEN), dtype=np.uint8)
    P = np.zeros(MAX_LEN, dtype=np.float32)

    temp_A = Chem.rdmolops.GetAdjacencyMatrix(mol).astype(np.uint8, copy=False)[:MAX_LEN, :MAX_LEN]
    num_atom = temp_A.shape[0]
    A[:num_atom, :num_atom] = temp_A + np.eye(temp_A.shape[0], dtype=np.uint8)

    for i, atom in enumerate(mol.GetAtoms()):
        feature = atom_feature(atom)
        X[i, :] = feature
        P[i] = LIST_PROB[feature[0]]
        if i + 1 >= num_atom: break
    P /= P.sum()

    return X, A, P


def preprocess_df(smiles, num_worker):
    with mp.Pool(processes=num_worker) as pool:
        mols = pool.map(mol2graph, smiles)
    X, A, P = list(zip(*mols))
    X = np.array(X, dtype=np.uint8)
    A = np.array(A, dtype=np.uint8)
    P = np.array(P, dtype=np.float32)
    return X, A, P


class zincDataset(Dataset):
    def __init__(self, data_path, filename, num_worker, save_cache=True, labels=['logP', 'mr', 'tpsa']):

        # Find whether cache is exist
        files = get_dir_files(data_path)
        cache_name = filename + '.npz'
        if cache_name in files:
            print("Cache Found. Loading Preprocessed Data from {}...".format(cache_name))
            temp = np.load(join(data_path, cache_name))
            self.X = temp['X']
            self.A = temp['A']
            self.C = temp['C']
            self.P = temp['P']
            print("Loading Preprocessed Data Complete!".format(cache_name))

        else:
            print("Cache Not Found. Loading Dataset from {}...".format(filename))
            # Load data from raw dataset
            self.data = pd.read_csv(join(data_path, filename))
            self.data = self.data.reset_index()
            print("Dataset Loading Complete")

            # Mean & Std Normalize of molecular property
            self.data.logP = (self.data.logP - LOGP_MEAN) / LOGP_STD
            self.data.mr = np.log10(self.data.mr + 1)
            self.data.mr = (self.data.mr - MR_MEAN) / MR_STD
            self.data.tpsa= np.log10(self.data.tpsa + 1)
            self.data.tpsa = (self.data.tpsa - TPSA_MEAN) / TPSA_STD
            print("Molecular Property Normalization Complete!")

            # Get Property Matrix
            self.C = self.data[labels].values
            smiles = self.data.smile.values
            del self.data

            # Convert smiles to Graph
            print("Converting Smiles to Graph...")
            self.X, self.A, self.P = preprocess_df(smiles, num_worker)
            del smiles
            print("Converting Smiles to Graph Complete!")

            # Save Preprocessed Data
            if save_cache:
                print("Saving Preprocessed Data to {}...".format(cache_name))
                np.savez_compressed(join(data_path, filename), X=self.X, A=self.A, C=self.C, P=self.P)
                print("Saving Preprocessed Data Complete!")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        A = self.A[index]
        C = self.C[index]
        P = self.P[index]
        num_atom = len(X)
        return X, A, C, P, num_atom


if __name__ == '__main__':
    a = './dataset/data_xs/train/train000000.csv'
    dataset = zincDataset(a)