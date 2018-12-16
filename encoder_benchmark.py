
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
import numpy as np

# Source code for fc models (tf for classification, tf_regression for regression)
# https://github.com/deepchem/deepchem/blob/master/deepchem/models/tensorgraph/fcnet.py


# In[ ]:

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
def benchmark(dir_path, ckpt_file, tasks, _model, featurizer, metric_type, hps):
    
    if featurizer == 'comet':
        checkpoint = torch.load(join(dir_path, ckpt_file), map_location=torch.device('cuda'))
        args = checkpoint['args']
        _args = vars(args)
        args.batch_size = 1
        args.test_batch_size = 1
        args.act = 'gelu'
        comet = model.Encoder(args)
        # comet = model_old.Encoder(args)
        comet.load_state_dict(checkpoint['encoder'])
        comet.eval()

    elif featurizer == 'rand':
        _args = {'input': 'rand'}

    def mol_to_graph(mol):

        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        list_feature = list()
        for atom in mol.GetAtoms():
            list_feature.append(atom_feature(atom))
        return np.array(list_feature), adj
    
    class RandFeat(Featurizer):
        name = ['random_featurizer']
        def __init__(self, dim):
            self.dim = dim
            
        def _featurize(self, batch):
            return np.random.rand(self.dim)

    class Comet(Featurizer):
        name = ['comet_encoder']
        def __init__(self, model):
            self.device = 'cuda'

            self.model = model.to(self.device)

        def _featurize(self, batch):
            X, A = batch
            X = Variable(torch.from_numpy(X).to(self.device).long())
            A = Variable(torch.from_numpy(A).to(self.device).float())
            _, _, molvec = self.model(X, A)
            return molvec.detach().cpu().numpy()


    if featurizer == 'comet':
        feat = Comet(comet)
    if featurizer == 'raw':
        feat = dc.feat.raw_featurizer.RawFeaturizer()
    if featurizer == 'fingerprint':
        feat = dc.feat.fingerprints.CircularFingerprint(size=1024)
    if featurizer == 'default':
        feat = None
    if featurizer == 'rand':
        feat = RandFeat(512)
    
    for task in tasks:
        if metric_type == 'reg':
            _metric = get_reg_metric(task)
        elif metric_type == 'cls':
            _metric = [dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)]
        task_result = dc.molnet.run_benchmark(
                                     ckpt = ckpt_file,
                                     arg = _args,
                                     datasets = [task],
                                     model = _model,
                                     split = None,
                                     metric = _metric,
                                     n_features = 512,
                                     featurizer = feat,
                                     out_path= './results',
                                     hyper_parameters = hps,
                                     test = True,
                                     reload = False,
                                     seed = 123
                                     )


cls_tasks = ['pcba', 'hiv'] #['bace_c',  'bbbp', 'tox21', 'toxcast', 'sider', 'clintox'] #['pcba', 'hiv']
reg_tasks =  ['pdbbind']  #['sampl', 'bace_r', 'delaney', 'hopv', 'lipo', 'pdbbind', 'ppb', 'qm7'] #'nci', 'chembl',  'qm8', 'qm9',
# Take Too Long 'kaggle'
# Shape issue 'qm7b'

# reg_models = ['tf_regression', 'tf_regression_ft', 'rf_regression', 'graphconvreg',
#       'dtnn', 'dag_regression', 'xgb_regression', 'weave_regression', 'krr', 'ani']

# cls_tasks = ['bace_c', 'bbbp', 'tox21', 'sider', 'clintox']
# reg_tasks = ['qm7', 'qm8', 'lipo', 'qm7', 'qm8', 'delaney', 'sampl']


def get_reg_metric(task):
    if task in ['sampl', 'lipo', 'pdbbind', 'delaney']:
        return [dc.metrics.Metric(dc.metrics.rms_score, np.mean, mode='regression')]

    else: # task in ['qm7', 'qm8', 'qm9']:
        return [dc.metrics.Metric(dc.metrics.mae_score, np.mean, mode='regression')]

"""
for task in reg_tasks:
    metric = get_reg_metric(task)
    temp_hps = hps['tf_regression']
    temp_hps.update({'batch_size':256}) #, 'nb_epoch':30})
    fingerprint_result = dc.molnet.run_benchmark(
                                     ckpt = 'fingerprint',
                                     arg = {'input':'fingerprint'},
                                     datasets = [task],
                                     model = 'tf_regression',
                                     split = None,
                                     metric =  metric,
                                     n_features = 1024,
                                     featurizer = dc.feat.fingerprints.CircularFingerprint(size=1024),
                                     out_path= './results',
                                     hyper_parameters = temp_hps,
                                     # hyper_param_search=True,
                                     test = True,
                                     reload = False,
                                     seed = 123
                                     )


temp_hps = hps['tf']
temp_hps.update({'batch_size':256}) #, 'nb_epoch':30})
fingerprint_result = dc.molnet.run_benchmark(
                                 ckpt = 'fingerprint',
                                 arg = {'input':'fingerprint'},
                                 datasets = cls_tasks,
                                 model = 'tf',
                                 split = None,
                                 metric =[dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)],
                                 n_features = 512,
                                 featurizer = dc.feat.fingerprints.CircularFingerprint(size=512),
                                 out_path= './results',
                                 hyper_parameters = temp_hps,
                                 test = True,
                                 reload = False,
                                 seed = 123
                                 )

for task in reg_tasks:
    metric = get_reg_metric(task)
    temp_hps = hps['tf_regression']
    temp_hps.update({'batch_size':256}) #, 'nb_epoch':30})
    fingerprint_result = dc.molnet.run_benchmark(
                                     ckpt = 'rand',
                                     arg = {'input':'rand'},
                                     datasets = [task],
                                     model = 'tf_regression',
                                     split = None,
                                     metric =  metric,
                                     n_features = 1024,
                                     featurizer = 'rand',
                                     out_path= './results',
                                     hyper_parameters = temp_hps,
                                     # hyper_param_search=True,
                                     test = True,
                                     reload = False,
                                     seed = 123
                                     )
"""
"""
temp_hps = hps['tf']
temp_hps.update({'batch_size':256}) #, 'nb_epoch':30})
fingerprint_result = benchmark('', 'rand', cls_tasks, 'tf', 'rand', 'cls', hps=temp_hps)

temp_hps = hps['tf_regression']
temp_hps.update({'batch_size':256}) #, 'nb_epoch':30})
fingerprint_result = benchmark('', 'rand', reg_tasks, 'tf_regression', 'rand', 'reg', hps=temp_hps)
"""

def benchmark_dir(dir_path):
    list_file = [ file for file in get_dir_files(dir_path) if '.tar' in file]
    list_file.sort()
    idx = 0
    skip = 1
    while idx < len(list_file):
        try:
            ckpt = list_file[idx]
            print('####################################################################')
            print('benchmarking : {}.  Progress : {}/{}'.format(ckpt, idx, len(list_file)))
            print('####################################################################')

            # temp_hps = hps['tf']
            # temp_hps.update({'batch_size':256})
            # benchmark(dir_path, ckpt, cls_tasks, 'tf', 'comet', 'cls', hps=temp_hps)

            temp_hps = hps['tf_regression']
            temp_hps.update({'batch_size':256})
            benchmark(dir_path, ckpt, reg_tasks, 'tf_regression', 'comet', 'reg', hps=temp_hps)
        except:
            pass
        finally:
            idx += skip

if __name__ == '__main__':
    pass
    dir_path1 = './runs/exp2_l4_o256_v512_r1_lf0.2_train'
    benchmark_dir(dir_path1)
    dir_path2 = './runs/exp2_l4_o256_v512_r1_lf0.5_train'
    benchmark_dir(dir_path2)
    dir_path3 = './runs/exp1_l4_o256_v512_r1_train'
    benchmark_dir(dir_path3)

    # ===== For Rand ===== #
    """
    dir_path = './runs/rand'
    ckpt_model = 'rand'
    temp_hps = hps['tf_regression']
    temp_hps.update({'batch_size': 256})
    benchmark(dir_path, 'rand', reg_tasks, 'tf_regression', 'rand', 'reg', hps=temp_hps)
    """
    
