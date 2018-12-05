

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler
from torch._six import int_classes as _int_classes
from rdkit import Chem
from scipy.linalg import fractional_matrix_power
import numpy as np
import pandas as pd





def atom_feature(atom):
    return np.array(char_to_ix(atom.GetSymbol(),
                              ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                               'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                               'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                               'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (40, 6, 5, 6, 1)


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def char_to_ix(x, allowable_set):
    if x not in allowable_set:
        return [0] # Unknown Atom Token
    return [allowable_set.index(x)+1]


def random_onehot(size):
    """ Generate random one-hot encoding vector with given size. """
    temp = np.zeros(size)
    temp[np.random.randint(0, size)] = 1
    return temp 


def normalize_adj(mx):
    """ Symmetry Normalization """
    rowsum = np.diag(np.array(mx.sum(1)))
    r_inv = fractional_matrix_power(rowsum, -0.5)
    r_inv[np.isinf(r_inv)] = 0.
    return r_inv.dot(mx).dot(r_inv)


def masking_feature(feature, num_masking):
    """ Given feature, select 'num_masking' node feature and perturbate them.
    
        [5 features : Atom symbol, degree, num Hs, valence, isAromatic]  
        were masked with zero or changed with random one-hot encoding 
        or remained with origianl data(but still should be predicted).
        
        Masking process was conducted on each feature indiviually. 
        For example, if ERASE_RATE = 0.5, probability for all feature information with zero is 0.5^5 = 0.03125
        
        return original hode feature with their corresponding indices
    """
    MASKING_RATE = 0.15
    ERASE_RATE = 0.5
    
    masking_indices = np.random.choice(len(feature), num_masking, replace=False)
    ground_truth = np.copy(feature[masking_indices, :])
    for i in masking_indices:
        prob_masking = np.random.rand(5)
        # Masking Atom Symbol 
        if prob_masking[0] < ERASE_RATE:
            feature[i, 0] = 0
        elif prob_masking[0] > 1- ((1-ERASE_RATE) * 0.5):
            feature[i, 0] = np.random.randint(1, 41)
            
        # Masking Degree 
        if prob_masking[1] < ERASE_RATE:
            feature[i, 1:7] = np.zeros(6)
        elif prob_masking[1] > 1- ((1-ERASE_RATE) * 0.5):
            feature[i, 1:7] =  random_onehot(6)
        
        # Masking Num Hs
        if prob_masking[2] < ERASE_RATE:
            feature[i, 7:12] = np.zeros(5)
        elif prob_masking[2] > 1- ((1-ERASE_RATE) * 0.5):
            feature[i, 7:12] =  random_onehot(5)
            
        # Masking Valence
        if prob_masking[3] < ERASE_RATE:
            feature[i, 12:18] = np.zeros(6)
        elif prob_masking[3] > 1- ((1-ERASE_RATE) * 0.5):
            feature[i, 12:18] =  random_onehot(6)
            
        # Masking IsAromatic
        if prob_masking[4] < ERASE_RATE:
            feature[i, 18] = (feature[i, 18]+1)%2

    return feature, ground_truth, masking_indices


def postprocess_batch(mini_batch):
    
    MASKING_RATE = 0.15
    ERASE_RATE = 0.5
    """ Given mini-batch sample, adjacency matrix and node feature vectors were padded with zero. """
    max_length = int(max([row[0] for row in mini_batch]))
    min_length = int(min([row[0] for row in mini_batch]))
    num_masking = max(1, int(max_length * MASKING_RATE))
    batch_length = len(mini_batch)
    batch_feature = np.zeros((batch_length, max_length, mini_batch[0][1].shape[1]), dtype=int)
    batch_adj = np.zeros((batch_length, max_length, max_length))
    batch_property = np.zeros((batch_length, 3))
    batch_ground = np.zeros((batch_length, num_masking, mini_batch[0][1].shape[1]), dtype=int)
    batch_masking = np.zeros((batch_length, num_masking), dtype=int)
    
    for i, row in enumerate(mini_batch):
        mol_length, feature, adj = int(row[0]), row[1], row[2]
        masked_feature, ground_truth, masking_indices  = masking_feature(feature, num_masking)
        batch_feature[i, :mol_length, :] = masked_feature
        batch_ground[i, :, :] = ground_truth
        batch_masking[i, :] = masking_indices
        batch_adj[i, :mol_length, :mol_length] = normalize_adj(adj+np.eye(len(adj)))
        batch_property[i, :] = [row[3], row[4], row[5]]
        
    return batch_feature, batch_adj, batch_property, batch_ground, batch_masking


class BatchSampler(Sampler):

    def __init__(self, sampler, batch_size, drop_last=False, shuffle_batch=False):

        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or                 batch_size <= 0:
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
        self.data = self.data.sort_values(by=['length'])
        self.data = self.data.reset_index()
        self.data['mr'] = np.log10(self.data['mr'] + 1)
        self.data['tpsa'] = np.log10(self.data['tpsa'] + 1)
        self.data = self.data.to_dict('index')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        smile = row['smile']

        mol = Chem.MolFromSmiles(smile)
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        list_feature = list()
        for atom in mol.GetAtoms():
            list_feature.append(atom_feature(atom))

        return row['length'], np.array(list_feature), adj, row['logP'], row['mr'], row['tpsa']

    def get_sizes(self):
        return self.data['length']
    
class zincDataLoader(DataLoader):
    def __init__(self, data_path, batch_size, drop_last, shuffle_batch, num_workers):
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