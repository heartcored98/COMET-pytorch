from decimal import Decimal
import datetime
import os
from os.path import isfile, join

def make_model_comment(args, prior_keyword=('num_layers', 'out_dim',
                                            'molvec_dim', 'sc_type',
                                            'use_attn', 'n_attn_heads',
                                            'use_bn', 'emb_train',
                                            'train_logp', 'train_mr',
                                            'train_tpsa', 'optim',
                                            'lr', 'l2_coef',
                                            'dp_rate', 'batch_size',
                                            'train_logp', 'train_mr', 'train_tpsa')):
    model_name = datetime.datetime.now().strftime('%y-%m-%d_%H:%M:%S') + "_"
    dict_args = vars(args)
    if 'bar' in dict_args:
        del dict_args['bar']
    for keyword in prior_keyword:
        value = str(dict_args[keyword])
        if value.isdigit():
            try:
                value = int(value)
                model_name += keyword + ':{}_'.format(dict_args[keyword])
            except:
                model_name += keyword + ':{:.2E}_'.format(Decimal(dict_args[keyword]))
        else:
            model_name += keyword + ':{}_'.format(value)
    return model_name[:254]


def get_dir_files(dir_path):
    list_file = [f for f in os.listdir(dir_path) if isfile(join(dir_path, f))]
    return list_file