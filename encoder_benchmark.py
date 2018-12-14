
# coding: utf-8

# In[3]:


# import os
# os.system('export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}')
# os.environ['LD_LIBRARY_PATH']='/usr/local/cuda-9.0/lib64 {LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'


# In[4]:


from torch.autograd import Variable
import deepchem as dc
from deepchem.feat import Featurizer
from deepchem.molnet.preset_hyper_parameters import hps
from dataloader import *
import model
import model_old
from os import listdir
from os.path import isfile, join
from pprint import pprint
from tqdm import tqdm
import tensorflow as tf

# Source code for fc models (tf for classification, tf_regression for regression)
# https://github.com/deepchem/deepchem/blob/master/deepchem/models/tensorgraph/fcnet.py


# In[ ]:

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
def benchmark(dir_path, ckpt_file, tasks, _model, featurizer, metric_type):
    
    header = ckpt_file.split('.')[0]
    _ckpt = header+'_'+featurizer
    checkpoint = torch.load(join(dir_path, ckpt_file), map_location=torch.device('cuda'))
    args = checkpoint['args']
    _args = vars(args)
    args.batch_size = 1
    args.test_batch_size = 1
    args.act = 'gelu'
#     comet = model.Encoder(args)
    comet = model_old.Encoder(args)
    comet.load_state_dict(checkpoint['encoder'])
    comet.eval()

    def mol_to_graph(mol):

        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        list_feature = list()
        for atom in mol.GetAtoms():
            list_feature.append(atom_feature(atom))
        return np.array(list_feature), adj

    class MyEncoder(Featurizer):
        name = ['comet_encoder']
        def __init__(self, model):
            self.device = 'cuda'

            self.model = model.to(self.device)

        def _featurize(self, mol):
            X, A = mol_to_graph(mol)
            X = Variable(torch.unsqueeze(torch.from_numpy(X), dim=0)).to(self.device).long()
            A = Variable(torch.unsqueeze(torch.from_numpy(A.astype(float)), dim=0)).to(self.device).float()
            _, _, molvec = self.model(X, A)
            return torch.squeeze(molvec).detach().cpu().numpy()

    if metric_type == 'reg':
        _metric = [dc.metrics.Metric(dc.metrics.mae_score, np.mean)]
    if metric_type == 'cls':
        _metric = [dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)]
    
    if featurizer == 'comet':
        feat = MyEncoder(comet)
    if featurizer == 'raw':
        feat = dc.feat.raw_featurizer.RawFeaturizer()
    if featurizer == 'fingerprint':
        feat = dc.feat.fingerprints.CircularFingerprint(size=256)
    if featurizer == 'default':
        feat = None
    

    task_result = dc.molnet.run_benchmark(
                                 ckpt = _ckpt,
                                 arg = _args,
                                 datasets = tasks,
                                 model = _model,
                                 split = None,
                                 metric = _metric,
                                 n_features = 256,
                                 featurizer = feat,
                                 out_path= './results',
                                 hyper_parameters = None,
                                 test = True,
                                 reload = True,
                                 seed = 123
                                 )


# ## task types
# ---
# 1) classification: muv, hiv, bace_c, bbbp, tox21, toxcast, sider, clintox
# 
# 
# 2) regression: lipo, qm7, qm8, delaney, sampl

cls_tasks = [ 'hiv', 'bace_c',  'bbbp', 'tox21', 'toxcast', 'sider', 'clintox'] #'pcba',
# Dataset issue 'muv',

# 
reg_tasks =  [ 'qm8', 'qm9', 'sampl', 'bace_r', 'delaney', 'hopv', 'lipo', 'pdbbind', 'ppb', 'qm7'] #'nci', 'chembl'
# Take Too Long 'kaggle'
# Shape issue 'qm7b'


# reg_models = ['tf_regression', 'tf_regression_ft', 'rf_regression', 'graphconvreg',
#       'dtnn', 'dag_regression', 'xgb_regression', 'weave_regression', 'krr', 'ani']



# cls_tasks = ['bace_c', 'bbbp', 'tox21', 'sider', 'clintox']
# reg_tasks = ['qm7', 'qm8', 'lipo', 'qm7', 'qm8', 'delaney', 'sampl']



# ## Fingerprint

# In[ ]:

"""
fingerprint_result = dc.molnet.run_benchmark(
                                 ckpt = 'fingerprint',
                                 arg = {'input':'fingerprint'},
                                 datasets = reg_tasks,
                                 model = 'tf_regression',
                                 split = None,
                                 metric =  [dc.metrics.Metric(dc.metrics.mae_score, np.mean)],
                                 n_features = 256,
                                 featurizer = dc.feat.fingerprints.CircularFingerprint(size=256),
                                 out_path= './benchmark',
                                 hyper_parameters = hps['tf_regression'].update({'batch_size':2048}),
                                 # hyper_param_search=True,
                                 test = True,
                                 reload = False,
                                 seed = 123
                                 )

"""
# In[ ]:


fingerprint_result = dc.molnet.run_benchmark(
                                 ckpt = 'fingerprint',
                                 arg = {'input':'fingerprint'},
                                 datasets = cls_tasks,
                                 model = 'tf',
                                 split = None,
                                 metric = None,
                                 n_features = 256,
                                 featurizer = dc.feat.fingerprints.CircularFingerprint(size=256),
                                 out_path= './benchmark/',
                                 hyper_parameters = hps['tf'].update({'batch_size':2048}),
                                 test = True,
                                 reload = False,
                                 seed = 123
                                 )

