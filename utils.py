from decimal import Decimal
import datetime
import os
from os.path import isfile, join
import logging
import torch

def make_model_comment(args):
    model_explain = "Time : {} \n".format(datetime.datetime.now().strftime('%y-%m-%d_%H:%M:%S'))
    dict_args = vars(args)
    if 'bar' in dict_args:
        del dict_args['bar']
    for keyword, value in dict_args.items():
        value = str(dict_args[keyword])
        if value.isdigit():
            try:
                value = int(value)
                model_explain += keyword + ':{}_'.format(dict_args[keyword]) + '\n'
            except:
                model_explain += keyword + ':{:.2E}_'.format(Decimal(dict_args[keyword])) + '\n'
        else:
            model_explain += keyword + ':{}_'.format(value) + '\n'
    return model_explain


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
        'optimizer': optimizer.state_dict(),
        'args': args
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


def log_histogram(models, writer, cnt_iter):
    encoder = models['encoder']
    for name, param in encoder.named_parameters():
        name = name.replace('.', '/')

        idx1, idx2 = name.split('/')[-1], name.split('/')[-2]
        if idx1.isdigit():
            if int(idx1) > 0:
                continue
        if idx2.isdigit():
            if int(idx2) > 0:
                continue

        idx = name.find('attn')
        if idx > 0:
            name = name[idx:]
        writer.add_histogram(name, param.clone().cpu().data.numpy(), cnt_iter)

    classifier = models['classifier']
    for name, param in classifier.named_parameters():
        name = 'classifier/' + name
        writer.add_histogram(name, param.clone().cpu().data.numpy(), cnt_iter)

    if 'logP' in models:
        for name, param in models['logP'].named_parameters():
            name = 'logP/' + name
            writer.add_histogram(name, param.clone().cpu().data.numpy(), cnt_iter)

    if 'mr' in models:
        for name, param in models['mr'].named_parameters():
            name = 'mr/' + name
            writer.add_histogram(name, param.clone().cpu().data.numpy(), cnt_iter)

    if 'tpsa' in models:
        for name, param in models['tpsa'].named_parameters():
            name = 'tpsa/' + name
            writer.add_histogram(name, param.clone().cpu().data.numpy(), cnt_iter)
