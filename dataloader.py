

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler
from torch._six import int_classes as _int_classes
from rdkit import Chem
from scipy.linalg import fractional_matrix_power
import numpy as np
import pandas as pd


MASKING_RATE = 0
ERASE_RATE = 0
MAX_LEN = 0

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


def mol2graph(smi):
    mol = Chem.MolFromSmiles(smi)

    X = np.zeros((max_len, 5), dtype=np.float32)
    A = np.zeros((max_len, max_len), dtype=np.int8)
    P = np.zeros(max_len, dtype=np.float32)

    temp_A = Chem.rdmolops.GetAdjacencyMatrix(mol).astype(np.int8, copy=False)[:max_len, :max_len]
    num_mol = temp_A.shape[0]

    A[:num_mol, :num_mol] = temp_A + np.eye(temp_A.shape[0], dtype=np.int8)

    for i, atom in enumerate(mol.GetAtoms()):
        feature = atom_feature(atom)
        X[i, :] = feature
        P[i] = LIST_PROB[feature[0]]
        if i + 1 >= num_mol: break
    return X, A, P

def preprocess_df(smiles, num_worker):
    with mp.Pool(processes=num_worker) as pool:
        mols = pool.map(mol2graph, smiles)
    print(len(mols))

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


def postprocess_batch(mini_batch):
    # Assign masking and erase rate from global variables
    masking_rate = MASKING_RATE
    erase_rate = ERASE_RATE

    """ Given mini-batch sample, adjacency matrix and node feature vectors were padded with zero. """
    max_length = int(max([row[0] for row in mini_batch]))
    min_length = int(min([row[0] for row in mini_batch]))
    num_masking = max(1, int(min_length * masking_rate))
    batch_length = len(mini_batch)
    batch_masked_feature = np.zeros((batch_length, max_length, mini_batch[0][1].shape[1]), dtype=int)
    batch_original_feature = np.zeros((batch_length, max_length, mini_batch[0][1].shape[1]), dtype=int)
    batch_adj = np.zeros((batch_length, max_length, max_length))
    batch_property = np.zeros((batch_length, 3))
    batch_ground = np.zeros((batch_length, num_masking, mini_batch[0][1].shape[1]), dtype=int)
    batch_masking = np.zeros((batch_length, num_masking), dtype=int)
    
    for i, row in enumerate(mini_batch):
        mol_length, original_feature, adj, list_prob = int(row[0]), row[1], row[2], row[3]
        masked_feature, ground_truth, masking_indices  = masking_feature(original_feature, num_masking, erase_rate, list_prob)
        batch_masked_feature[i, :mol_length, :] = masked_feature
        batch_original_feature[i, :mol_length, :] = original_feature
        batch_ground[i, :, :] = ground_truth
        batch_masking[i, :] = masking_indices
        batch_adj[i, :mol_length, :mol_length] = adj #normalize_adj(adj+np.eye(len(adj)))
        batch_property[i, :] = [row[4], row[5], row[6]]
        
    return batch_original_feature, batch_masked_feature, batch_adj, batch_property, batch_ground, batch_masking


class BatchSampler(Sampler):

    def __init__(self, sampler, batch_size, drop_last=False, shuffle_batch=False):

        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_batch = shuffle_batch

    def __iter__(self):
        batch = list()
        mini_batch = list()
        for idx in self.sampler:
            mini_batch.append(idx)
            if len(mini_batch) == self.batch_size:
                batch.append(mini_batch)
                mini_batch = []
        if len(mini_batch) > 0 and not self.drop_last:
            batch.append(mini_batch)
            
        if self.shuffle_batch:
            return iter(np.random.permutation(batch))
        else:
            return iter(batch)
            

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class zincDataset(Dataset):
    def __init__(self, data_path, skip_header=True):
        self.data = pd.read_csv(data_path)
        self.data = self.data.reset_index()
        # self.data = self.data.sort_values(by=['length'])

        # Mean & Std Normalize of molecular property
        self.data.logP = (self.data.logP - LOGP_MEAN) / LOGP_STD
        self.data.mr = np.log10(self.data.mr + 1)
        self.data.mr = (self.data.mr - MR_MEAN) / MR_STD
        self.data.tpsa= np.log10(self.data.tpsa + 1)
        self.data.tpsa = (self.data.tpsa - TPSA_MEAN) / TPSA_STD

        # self.data = self.data.to_dict('index')

    def get_df(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        smile = row['smile']

        mol = Chem.MolFromSmiles(smile)
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)

        adj = normalize_adj(adj + np.eye(len(adj)))

        num_mol = len(mol.GetAtoms())
        list_feature = np.zeros((num_mol, 5))
        list_prob = np.zeros(num_mol)
        for i, atom in enumerate(mol.GetAtoms()):
            feature = atom_feature(atom)
            list_feature[i] = feature
            list_prob[i] = LIST_PROB[feature[0]]

        return row['length'], list_feature, adj, list_prob, row['logP'], row['mr'], row['tpsa']

    def get_sizes(self):
        return self.data['length']
    
class zincDataLoader(DataLoader):
    def __init__(self, data_path, batch_size, drop_last, shuffle_batch, num_workers, masking_rate, erase_rate):
        global MASKING_RATE, ERASE_RATE
        MASKING_RATE = masking_rate
        ERASE_RATE = erase_rate

        train_dataset = zincDataset(data_path=data_path)
        sampler = SequentialSampler(train_dataset)
        SortedBatchSampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=drop_last, shuffle_batch=shuffle_batch)
        DataLoader.__init__(self, train_dataset,
                            collate_fn=postprocess_batch, 
                            num_workers=num_workers, 
                            batch_sampler=SortedBatchSampler,
                            pin_memory=True)

if __name__ == '__main__':
    a = './dataset/data_xs/train/train000000.csv'
    dataset = zincDataset(a)