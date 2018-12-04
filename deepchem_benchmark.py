

from torch.autograd import Variable
import deepchem as dc
from deepchem.feat import Featurizer

from dataloader import *
import model

# * Source code for fc models (tf for classification, tf_regression for regression)
# https://github.com/deepchem/deepchem/blob/master/deepchem/models/tensorgraph/fcnet.py

ckpt = './model/model_ck_004_000016000.tar'

def benchmark(ckpt_file):
    # load checkpoint file 
    checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
    args = checkpoint['args']
    args.batch_size = 1
    args.test_batch_size = 1
    comet = model.Encoder(args)
    comet.load_state_dict(checkpoint['encoder'])

    def mol_to_graph(mol):
        #print(mol)
        #mol = Chem.MolFromSmiles(mol)
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        list_feature = list()
        for atom in mol.GetAtoms():
            list_feature.append(atom_feature(atom))
        return np.array(list_feature), adj

    class MyEncoder(Featurizer):
        name = ['comet_encoder']
        def __init__(self, model):
            self.model = model

        def _featurize(self, mol):
            X, A = mol_to_graph(mol)
            X = Variable(torch.unsqueeze(torch.from_numpy(X), dim=0)).long()
            A = Variable(torch.unsqueeze(torch.from_numpy(A.astype(float)), dim=0)).float()
            _, _, molvec = self.model(X, A)
            return torch.squeeze(molvec)


    
    filename = args.model_name
    reg_path = './benchmark/'+'reg_'+filename+'.csv'
    cls_path = './benchmark/'+'cls_'+filename+'.csv'

    reg_tasks = dc.molnet.run_benchmark(
                                 datasets= ['bace_c', 'bbbp', 'clintox', 'hiv', 'muv', 'pcba', 'sider', 'tox21', 'toxcast'],
                                 model = 'tf', 
                                 split = None,
                                 metric = None,
                                 featurizer = MyEncoder(comet),
                                 out_path= reg_path,
                                 hyper_parameters = None,
                                 test = True,
                                 reload = False,
                                 seed = 123 )
    
    cls_tasks = dc.molnet.run_benchmark(
                                 datasets= ['bace_r', 'chembl', 'clearance', 'delaney', 'hopv', 'kaggle', 'lipo',
                                            'nci', 'pdbbind', 'ppb', 'qm7', 'qm7b', 'qm8', 'qm9', 'sampl'], 
                                 model = 'tf_regression', 
                                 split = None,
                                 metric = None,
                                 featurizer = MyEncoder(comet),
                                 out_path= cls_path,
                                 hyper_parameters = None,
                                 test = True,
                                 reload = False,
                                 seed = 123 )
    
    return reg_tasks, cls_tasks
            

benchmark(ckpt)
