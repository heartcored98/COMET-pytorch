from decimal import Decimal
import datetime
import os
from os.path import isfile, join
import logging
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt


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


def save_checkpoint(epoch, cnt_iter, models, optimizers, args):
    checkpoint = {
        'epoch': epoch,
        'cnt_iter': cnt_iter,
        'args': args
    }
    for model_name, model in models.items():
        checkpoint.update({model_name: model.state_dict()})

    for optimizer_name, optimizer in optimizers.items():
        checkpoint.update({optimizer_name: optimizer.state_dict()})

    log_path = join(args.log_path, args.model_name + '_train')
    filename = '{}_{:03}_{:09}.tar'.format(args.model_name, epoch, cnt_iter)
    path = join(log_path, filename)
    torch.save(checkpoint, path)
    return filename


def load_checkpoint(models, optimizers, filename, args):
    log_path = join(args.log_path, args.model_name + '_train')
    checkpoint = torch.load(join(log_path, filename))

    for model_name, model in models.items():
        model.load_state_dict(checkpoint[model_name])


    for optimizer_name, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint[optimizer_name])

    return checkpoint['epoch'], checkpoint['cnt_iter'], models, optimizers


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


def plot_confusion_matrix(cm, labels, classes=None, title='Confusion matrix', normalize=True, figsize=(3,3)):
    if normalize:
        de = cm.sum(axis=1)[:, np.newaxis]
        de[de == 0] =1
        cm = cm.astype('float') / de

    np.set_printoptions(precision=2)

    fig = plt.Figure(figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    if classes:
        labels = classes
    tick_marks = np.arange(len(labels))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=6, rotation=0,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=6, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    ax.set_title(title, fontdict={'fontsize': 6})

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt) if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    return fig

def f1_macro(cm):
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    de1 = (TP+FP)
    de1[de1 == 0] = 1
    de2 =  (TP + FN)
    de2[de2 == 0] = 1
    precision = TP / de1
    recall = TP / de2
    de3 = (precision + recall)
    de3[de3 == 0] = 1
    f1 = 2 * (precision * recall) / de3
    macro_f1 = np.mean(f1)
    return macro_f1