from decimal import Decimal
import datetime
import os
from os.path import isfile, join
import logging
import torch

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


def get_logger(log_path, filename='train.log', logger_name=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(join(log_path, 'train.log'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_checkpoint(epoch, cnt_iter, models, optimizer, args):
    checkpoint = {
        'epoch': epoch,
        'cnt_iter': cnt_iter,
        'optimizer': optimizer.state_dict()
    }
    for model_name, model in models.items():
        checkpoint.update({model_name: model.state_dict()})

    log_path = join(args.log_path, args.model_name + '_train')
    filename = 'model_ck_{:03}_{:09}.tar'.format(epoch, cnt_iter)
    path = join(log_path, filename)
    torch.save(checkpoint, path)
    return filename


def load_checkpoint(models, optimizer, filename, args):
    log_path = join(args.log_path, args.model_name + '_train')
    checkpoint = torch.load(join(log_path, filename))

    for model_name, model in models.items():
        model.load_state_dict(checkpoint[model_name])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch'], checkpoint['cnt_iter'], models, optimizer

