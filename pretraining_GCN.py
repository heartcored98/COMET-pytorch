import time
import argparse

from tensorboardX import SummaryWriter
from sklearn.metrics import mean_absolute_error

from dataloader import *
from utils import *
from model import *
from metric import *


#######################
#===== Optimizer =====#
#######################


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self, step):
        "Update parameters and rate"
        rate = self.rate(step)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        step += 1
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def state_dict(self):
        return self.optimizer.state_dict()


########################
#===== Training   =====#
########################


def train(models, optimizer, dataloader, epoch, cnt_iter, args):
    t = time.time()
    list_train_loss = list()
    epoch = epoch
    cnt_iter = cnt_iter
    reg_loss = nn.MSELoss()

    for epoch in range(epoch, args.epoch+1):
        for batch_idx, batch in enumerate(dataloader['train']):
            t1 = time.time()
            # Setting Train Mode
            for _, model in models.items():
                model.train()

            optimizer['mask'].zero_grad()
            optimizer['auxiliary'].zero_grad()

            # Get Batch Sample from DataLoader
            num_masking = int(args.masking_rate * args.max_len)
            X, A, C, P = batch

            # Sampling Masking Center Atom
            center_idx = np.zeros(args.batch_size, dtype=np.uint8)
            for i, p_row in enumerate(P.numpy()):
                center_idx[i] = np.random.choice(np.array(args.max_len), 1, p=p_row)
            radius_A = torch.matrix_power(A.float(), args.radius)

            # Find Out which atom is connected to the center atom
            adjacent_A = torch.stack([adj[center_idx[i]] for i, adj in enumerate(radius_A)]) + 1e-6
            predict_idx = np.zeros((args.batch_size, num_masking), dtype=np.uint8)
            for i, p_row in enumerate(adjacent_A.numpy()):
                predict_idx[i] = np.random.choice(np.array(args.max_len), num_masking, p=p_row / p_row.sum(),
                                                  replace=False)

            # Get Target True X
            predict_idx = torch.Tensor(predict_idx).long()
            idx_1 = torch.LongTensor(range(args.batch_size)).unsqueeze(1).view(args.batch_size, -1).expand(-1, num_masking).flatten()
            true_X = X[idx_1, predict_idx.flatten(), :]

            # Get Input Masked X
            idx_2 = np.random.choice(np.array(args.batch_size), int(args.batch_size * args.erase_rate), replace=False)
            masking_idx = predict_idx[idx_2]
            idx_2 = torch.LongTensor(idx_2).unsqueeze(1).view(idx_2.shape[0], -1).expand(-1, num_masking).flatten()
            mask_X = torch.clone(X)
            mask_X[idx_2, masking_idx.flatten(), :] = 0

            # Normalize A matrix in order to prevent overflow

            # Convert Tensor into Variable and Move to CUDA
            mask_idx = Variable(predict_idx).to(args.device).long()
            input_X = Variable(X).to(args.device).long()
            mask_X = Variable(mask_X).to(args.device).long()
            true_X = Variable(true_X).to(args.device).long()
            input_A = Variable(A).to(args.device).float()
            #     mask_A = Variable(mask_A).to(args.device).float()
            input_C = Variable(C).to(args.device).float()
            logp, mr, tpsa = input_C[:, 0], input_C[:, 1], input_C[:, 2]

            t2 = time.time()
            # Encoding Masked Molecule
            encoded_X, _, molvec = models['encoder'](mask_X, input_A)
            pred_X = models['classifier'](encoded_X, molvec, mask_idx)

            # Compute Mask Task Loss
            symbol_loss, degree_loss, numH_loss, valence_loss, isarom_loss = compute_loss(pred_X, true_X)
            mask_loss = symbol_loss + degree_loss + numH_loss + valence_loss + isarom_loss

            # Backprogating and Updating Parameter
            mask_loss.backward()
            optimizer['mask'].step(cnt_iter)
            train_writer.add_scalar('1.status/lr', optimizer['mask'].rate(cnt_iter), cnt_iter)
            torch.cuda.empty_cache()

            t3 = time.time()
            # Encoding Original Molecule
            auxiliary_loss = None
            if args.train_logp or args.train_mr or args.train_tpsa:
                _, _, molvec = models['encoder'](input_X, input_A)

            # Compute Loss of Original Molecule Property
            if args.train_logp:
                pred_logp = models['logP'](molvec)
                logP_loss = reg_loss(pred_logp, logp)
                auxiliary_loss = logP_loss
            if args.train_mr:
                pred_mr = models['mr'](molvec)
                mr_loss = reg_loss(pred_mr, mr)
                auxiliary_loss = auxiliary_loss + mr_loss if auxiliary_loss else mr_loss
            if args.train_tpsa:
                pred_tpsa = models['tpsa'](molvec)
                tpsa_loss = reg_loss(pred_tpsa, tpsa)
                auxiliary_loss = auxiliary_loss + tpsa_loss if auxiliary_loss else tpsa_loss

            if args.train_logp or args.train_mr or args.train_tpsa:
                train_writer.add_scalar('1.status/auxiliary', auxiliary_loss, cnt_iter)
                auxiliary_loss.backward()
                optimizer['auxiliary'].step(cnt_iter)
                torch.cuda.empty_cache()

            t4 = time.time()
            print("total {:2.2f}. Prepare {:2.2f}. Mask {:2.2f}. Aux {:2.2f}".format(t4-t1, t2-t1, t3-t2, t4-t3))
            cnt_iter += 1
            setattr(args, 'epoch_now', epoch)
            setattr(args, 'iter_now', cnt_iter)

            # Prompting Status
            if cnt_iter % args.log_every == 0:
                train_writer.add_scalar('2.mask/symbol', symbol_loss, cnt_iter)
                train_writer.add_scalar('2.mask/degree', degree_loss, cnt_iter)
                train_writer.add_scalar('2.mask/numH', numH_loss, cnt_iter)
                train_writer.add_scalar('2.mask/valence', valence_loss, cnt_iter)
                train_writer.add_scalar('2.mask/isarom', isarom_loss, cnt_iter)
                train_writer.add_scalar('1.status/mask', mask_loss, cnt_iter)

                if args.train_logp:
                    train_writer.add_scalar('3.auxiliary/logP', logP_loss, cnt_iter)
                if args.train_mr:
                    train_writer.add_scalar('3.auxiliary/mr', mr_loss, cnt_iter)
                if args.train_tpsa:
                    train_writer.add_scalar('3.auxiliary/tpsa', tpsa_loss, cnt_iter)

                output = "[TRAIN] E:{:3}. P:{:>2.1f}%. Loss:{:>9.3}. Mask Loss:{:>9.3}. {:4.1f} mol/sec. Iter:{:6}.  Elapsed:{:6.1f} sec."
                elapsed = time.time() - t
                process_speed = (args.batch_size * args.log_every) / elapsed
                output = output.format(epoch, batch_idx / len(dataloader['train']) * 100.0, mask_loss, auxiliary_loss, process_speed, cnt_iter, elapsed,)
                t = time.time()
                logger.info(output)

            # Validate Model
            if cnt_iter % args.validate_every == 0:
                optimizer['mask'].zero_grad()
                optimizer['auxiliary'].zero_grad()
                validate(models, dataloader['val'], args, cnt_iter=cnt_iter, epoch=epoch)
                t = time.time()

            # Save Model
            if cnt_iter % args.save_every == 0:
                filename = save_checkpoint(epoch, cnt_iter, models, optimizer, args)
                logger.info('Saved Model as {}'.format(filename))
            del batch
                
    logger.info('Training Completed')


######################################
# ===== Validating and Testing   =====#
######################################


def validate(models, data_loader, args, **kwargs):
    t = time.time()
    cnt_iter = kwargs['cnt_iter']
    epoch = kwargs['epoch']
    temp_iter = 0
    reg_loss = nn.MSELoss()

    # For Loss
    list_mask_loss = []
    list_symbol_loss = []
    list_degree_loss = []
    list_numH_loss = []
    list_valence_loss = []
    list_isarom_loss = []
    list_logP_loss = []
    list_mr_loss = []
    list_tpsa_loss = []

    # For Accuracy & MAE metric
    list_symbol_acc = []
    list_degree_acc = []
    list_numH_acc = []
    list_valence_acc = []
    list_isarom_acc = []
    list_logP_mae = []
    list_mr_mae = []
    list_tpsa_mae = []

    # For F1-Score Metric & Confusion Matrix
    confusion_symbol = np.zeros((args.vocab_size, args.vocab_size))
    confusion_degree = np.zeros((args.degree_size, args.degree_size))
    confusion_numH = np.zeros((args.numH_size, args.numH_size))
    confusion_valence = np.zeros((args.valence_size, args.valence_size))
    confusion_isarom = np.zeros((args.isarom_size, args.isarom_size))


    # Initialization Model with Evaluation Mode
    for _, model in models.items():
        model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Get Batch Sample from DataLoader
            num_masking = int(args.masking_rate * args.max_len)
            X, A, C, P = batch

            # Sampling Masking Center Atom
            center_idx = np.zeros(args.batch_size, dtype=np.uint8)
            for i, p_row in enumerate(P.numpy()):
                center_idx[i] = np.random.choice(np.array(args.max_len), 1, p=p_row)
            radius_A = torch.matrix_power(A.float(), args.radius)

            # Find Out which atom is connected to the center atom
            adjacent_A = torch.stack([adj[center_idx[i]] for i, adj in enumerate(radius_A)]) + 1e-6
            predict_idx = np.zeros((args.batch_size, num_masking), dtype=np.uint8)
            for i, p_row in enumerate(adjacent_A.numpy()):
                predict_idx[i] = np.random.choice(np.array(args.max_len), num_masking, p=p_row / p_row.sum(),
                                                  replace=False)

            # Get Target True X
            predict_idx = torch.Tensor(predict_idx).long()
            idx_1 = torch.LongTensor(range(args.batch_size)).unsqueeze(1).view(args.batch_size, -1).expand(-1, num_masking).flatten()
            true_X = X[idx_1, predict_idx.flatten(), :]

            # Get Input Masked X
            idx_2 = np.random.choice(np.array(args.batch_size), int(args.batch_size * args.erase_rate), replace=False)
            masking_idx = predict_idx[idx_2]
            idx_2 = torch.LongTensor(idx_2).unsqueeze(1).view(idx_2.shape[0], -1).expand(-1, num_masking).flatten()
            mask_X = torch.clone(X)
            mask_X[idx_2, masking_idx.flatten(), :] = 0

            # Normalize A matrix in order to prevent overflow

            # Convert Tensor into Variable and Move to CUDA
            mask_idx = Variable(predict_idx).to(args.device).long()
            input_X = Variable(X).to(args.device).long()
            mask_X = Variable(mask_X).to(args.device).long()
            true_X = Variable(true_X).to(args.device).long()
            input_A = Variable(A).to(args.device).float()
            #     mask_A = Variable(mask_A).to(args.device).float()
            input_C = Variable(C).to(args.device).float()
            logp, mr, tpsa = input_C[:, 0], input_C[:, 1], input_C[:, 2]

            # Encoding Masked Molecule
            encoded_X, _, molvec = models['encoder'](mask_X, input_A)
            pred_mask = models['classifier'](encoded_X, molvec, mask_idx)

            # Compute Mask Task Loss & Property Regression Loss
            symbol_loss, degree_loss, numH_loss, valence_loss, isarom_loss = compute_loss(pred_mask, true_X)

            list_symbol_loss.append(symbol_loss.item())
            list_degree_loss.append(degree_loss.item())
            list_numH_loss.append(numH_loss.item())
            list_valence_loss.append(valence_loss.item())
            list_isarom_loss.append(isarom_loss.item())
            list_mask_loss.append((symbol_loss + degree_loss + numH_loss + valence_loss + isarom_loss).item())

            # Compute Mask Task Accuracy & Property Regression MAE
            symbol_acc, degree_acc, numH_acc, valence_acc, isarom_acc = compute_metric(pred_mask, true_X)
            list_symbol_acc.append(symbol_acc)
            list_degree_acc.append(degree_acc)
            list_numH_acc.append(numH_acc)
            list_valence_acc.append(valence_acc)
            list_isarom_acc.append(isarom_acc)

            # Accumulate Mask Task Confusion Matrix for F1-Metric
            confusions = compute_confusion(pred_mask, true_X, args)
            confusion_symbol += confusions[0]
            confusion_degree += confusions[1]
            confusion_numH += confusions[2]
            confusion_valence += confusions[3]
            confusion_isarom += confusions[4]

            if args.train_logp or args.train_mr or args.train_tpsa:
                _, _, molvec = models['encoder'](input_X, input_A)

            if args.train_logp:
                pred_logp = models['logP'](molvec)
                list_logP_loss.append(reg_loss(pred_logp, logp).item())
                list_logP_mae.append(mean_absolute_error(pred_logp.cpu().detach().numpy(), logp.cpu().detach().numpy()))
            if args.train_mr:
                pred_mr = models['mr'](molvec)
                list_mr_loss.append(reg_loss(pred_mr, mr).item())
                list_mr_mae.append(mean_absolute_error(pred_mr.cpu().detach().numpy(), mr.cpu().detach().numpy()))
            if args.train_tpsa:
                pred_tpsa = models['tpsa'](molvec)
                list_tpsa_loss.append(reg_loss(pred_tpsa, tpsa).item())
                list_tpsa_mae.append(mean_absolute_error(pred_tpsa.cpu().detach().numpy(), tpsa.cpu().detach().numpy()))
            temp_iter += 1

            # Prompting Status
            if temp_iter % (args.log_every * 10) == 0:
                output = "[VALID] E:{:3}. P:{:>2.1f}%. {:4.1f} mol/sec. Iter:{:6}.  Elapsed:{:6.1f} sec."
                elapsed = time.time() - t
                process_speed = (args.test_batch_size * args.log_every) / elapsed
                output = output.format(epoch, batch_idx / len(data_loader) * 100.0, process_speed, temp_iter, elapsed, )
                t = time.time()
                logger.info(output)
            del batch

    val_writer.add_figure('symbol/confusion',
                         plot_confusion_matrix(
                             confusion_symbol, range(args.vocab_size),
                             classes=LIST_SYMBOLS, title="Symbol CM @ {}".format(cnt_iter), figsize=(10, 10)),
                         cnt_iter)
    val_writer.add_figure('degree/confusion',
                         plot_confusion_matrix(confusion_degree, range(args.degree_size), title="Degree CM @ {}".format(cnt_iter)),
                         cnt_iter)
    val_writer.add_figure('numH/confusion',
                         plot_confusion_matrix(confusion_numH, range(args.numH_size), title="NumH CM @ {}".format(cnt_iter)),
                         cnt_iter)
    val_writer.add_figure('valence/confusion',
                         plot_confusion_matrix(confusion_valence, range(args.valence_size), title="Valence CM @ {}".format(cnt_iter)),
                         cnt_iter)
    val_writer.add_figure('isarom/confusion',
                         plot_confusion_matrix(confusion_isarom, range(args.isarom_size),
                                               title="isAromatic CM @ {}".format(cnt_iter), figsize=(2,2)),
                         cnt_iter)

    # Averaging Loss across the batch
    mask_loss = np.mean(np.array(list_mask_loss))
    symbol_loss = np.mean(np.array(list_symbol_loss))
    degree_loss = np.mean(np.array(list_degree_loss))
    numH_loss = np.mean(np.array(list_numH_loss))
    valence_loss = np.mean(np.array(list_valence_loss))
    isarom_loss = np.mean(np.array(list_isarom_loss))

    symbol_acc = np.mean(np.array(list_symbol_acc))
    degree_acc = np.mean(np.array(list_degree_acc))
    numH_acc = np.mean(np.array(list_numH_acc))
    valence_acc = np.mean(np.array(list_valence_acc))
    isarom_acc = np.mean(np.array(list_isarom_acc))

    val_writer.add_scalar('2.mask/symbol', symbol_loss, cnt_iter)
    val_writer.add_scalar('2.mask/degree', degree_loss, cnt_iter)
    val_writer.add_scalar('2.mask/numH', numH_loss, cnt_iter)
    val_writer.add_scalar('2.mask/valence', valence_loss, cnt_iter)
    val_writer.add_scalar('2.mask/isarom', isarom_loss, cnt_iter)

    val_writer.add_scalar('4.mask_metric/acc_symbol', symbol_acc, cnt_iter)
    val_writer.add_scalar('4.mask_metric/acc_degree', degree_acc, cnt_iter)
    val_writer.add_scalar('4.mask_metric/acc_numH', numH_acc, cnt_iter)
    val_writer.add_scalar('4.mask_metric/acc_valence', valence_acc, cnt_iter)
    val_writer.add_scalar('4.mask_metric/acc_isarom', isarom_acc, cnt_iter)

    val_writer.add_scalar('4.mask_metric/f1_symbol', f1_macro(confusion_symbol), cnt_iter)
    val_writer.add_scalar('4.mask_metric/f1_degree', f1_macro(confusion_degree), cnt_iter)
    val_writer.add_scalar('4.mask_metric/f1_numH', f1_macro(confusion_numH), cnt_iter)
    val_writer.add_scalar('4.mask_metric/f1_valence', f1_macro(confusion_valence), cnt_iter)
    val_writer.add_scalar('4.mask_metric/f1_isarom', f1_macro(confusion_isarom), cnt_iter)

    auxiliary_loss = None
    if args.train_logp:
        logP_loss = np.mean(np.array(list_logP_loss))
        logP_mae = np.mean(np.array(list_logP_mae))
        auxiliary_loss = logP_loss
        val_writer.add_scalar('3.auxiliary/logP', logP_loss, cnt_iter)
        val_writer.add_scalar('5.auxiliary_mae/logP', logP_mae, cnt_iter)
    if args.train_mr:
        mr_loss = np.mean(np.array(list_mr_loss))
        mr_mae = np.mean(np.array(list_mr_mae))
        auxiliary_loss = auxiliary_loss + mr_loss if auxiliary_loss else mr_loss
        val_writer.add_scalar('3.auxiliary/mr', mr_loss, cnt_iter)
        val_writer.add_scalar('5.auxiliary_mae/mr', mr_mae, cnt_iter)
    if args.train_tpsa:
        tpsa_loss = np.mean(np.array(list_tpsa_loss))
        tpsa_mae = np.mean(np.array(list_mr_mae))
        auxiliary_loss = auxiliary_loss + tpsa_loss if auxiliary_loss else tpsa_loss
        val_writer.add_scalar('3.auxiliary/tpsa', tpsa_loss, cnt_iter)
        val_writer.add_scalar('5.auxiliary_mae/tpsa', tpsa_mae, cnt_iter)
    if args.train_logp or args.train_mr or args.train_tpsa:
        val_writer.add_scalar('1.status/auxiliary', auxiliary_loss, cnt_iter)
    val_writer.add_scalar('1.status/mask', mask_loss, cnt_iter)

    # Log model weight historgram
    log_histogram(models, val_writer, cnt_iter)

    output = "[VALID] E:{:3}. P:{:>2.1f}%. Mask Loss:{:>9.3}. Aux Loss:{:>9.3}. {:4.1f} mol/sec. Iter:{:6}.  Elapsed:{:6.1f} sec."
    elapsed = time.time() - t
    process_speed = (args.test_batch_size * args.log_every) / elapsed
    output = output.format(epoch, batch_idx / len(data_loader) * 100.0, mask_loss, auxiliary_loss, process_speed, cnt_iter, elapsed)
    logger.info(output)
    torch.cuda.empty_cache()


def experiment(dataloader, args):
    ts = time.time()
    
    # Construct Model
    encoder = Encoder(args)
    classifier = Classifier(args.out_dim, args.molvec_dim, args.classifier_dim, args.dp_rate, ACT2FN[args.act])
    models = {'encoder': encoder, 'classifier': classifier}
    if args.train_logp:
        models.update({'logP': Regressor(args.molvec_dim, args.dp_rate, ACT2FN[args.act])})
    if args.train_mr:
        models.update({'mr': Regressor(args.molvec_dim, args.dp_rate, ACT2FN[args.act])})
    if args.train_tpsa:
        models.update({'tpsa': Regressor(args.molvec_dim, args.dp_rate, ACT2FN[args.act])})
        
    # Initialize Optimizer
    logger.info('####### Model Constructed #######')
    mask_trainable_parameters = list()
    auxiliary_trainable_parameters = list()
    for key, model in models.items():
        model.to(args.device)
        if key in ['encoder', 'classifier']:
            mask_trainable_parameters += list(filter(lambda p: p.requires_grad, model.parameters()))
        if key in ['encoder', 'logP', 'mr', 'tpsa']:
            auxiliary_trainable_parameters += list(filter(lambda p: p.requires_grad, model.parameters()))
        logger.info('{:10}: {:>10} parameters'.format(key, sum(p.numel() for p in model.parameters())))
        setattr(args, '{}_param'.format(key), sum(p.numel() for p in model.parameters()))
    logger.info('#################################')
    
    if args.optim == 'ADAM':
        mask_optimizer = optim.Adam(mask_trainable_parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
        auxiliary_optimizer = optim.Adam(auxiliary_trainable_parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
    elif args.optim == 'RMSProp':
        mask_optimizer = optim.RMSprop(mask_trainable_parameters, lr=0)
        auxiliary_optimizer = optim.RMSprop(auxiliary_trainable_parameters, lr=0)
    elif args.optim == 'SGD':
        mask_optimizer = optim.SGD(mask_trainable_parameters, lr=0)
        auxiliary_optimizer = optim.SGD(auxiliary_trainable_parameters, lr=0)
    else:
        assert False, "Undefined Optimizer Type"
    optimizers = {'mask':mask_optimizer, 'auxiliary':auxiliary_optimizer}

    # Reload Checkpoint Model
    epoch = 0
    cnt_iter = 0
    if args.ck_filename:
        epoch, cnt_iter, models, optimizers = load_checkpoint(models, optimizers, args.ck_filename, args)
        logger.info('Loaded Model from {}'.format(args.ck_filename))
    
    mask_optimizer = NoamOpt(args.out_dim, args.lr_factor, args.lr_step, optimizers['mask'])
    auxiliary_optimizer = NoamOpt(args.out_dim, args.lr_factor, args.lr_step, optimizers['auxiliary'])
    optimizers = {'mask':mask_optimizer, 'auxiliary':auxiliary_optimizer}

    # Train Model
    validate(models, dataloader['val'], args, cnt_iter=cnt_iter, epoch=epoch)
    train(models, optimizers, dataloader, epoch, cnt_iter, args)

    # Logging Experiment Result
    te = time.time()    
    args.elapsed = te-ts
    logger.info('Training Completed')

if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description='Add logP, TPSA, MR, PBF value on .smi files')

    # ===== Label Size Constant ===== #
    parser.add_argument("--vocab_size", type=int, default=40)
    parser.add_argument("--degree_size", type=int, default=6)
    parser.add_argument("--numH_size", type=int, default=5)
    parser.add_argument("--valence_size", type=int, default=6)
    parser.add_argument("--isarom_size", type=int, default=2)

    # ===== Model Definition =====#
    parser.add_argument("--in_dim", type=int, default=64)
    parser.add_argument("--classifier_dim", type=int, default=59)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--molvec_dim", type=int, default=256)

    parser.add_argument("-n", "--n_layer", type=int, default=4)
    parser.add_argument("-k", "--n_attn_heads", type=int, default=8)
    parser.add_argument("-c", "--sc_type", type=str, default='sc')

    parser.add_argument("-a", "--use_attn", type=bool, default=True)
    parser.add_argument("-b", "--use_bn", type=bool, default=True)
    parser.add_argument("-e", "--emb_train", type=bool, default=True)
    parser.add_argument("-dp", "--dp_rate", type=float, default=0.1)
    parser.add_argument("--act", type=str, default='gelu')

    # ===== Optimizer =====#
    parser.add_argument("-u", "--optim", type=str, default='ADAM')
    parser.add_argument("-lf", "--lr_factor", type=float, default=2.0)
    parser.add_argument("-ls", "--lr_step", type=int, default=4000)

    # ===== Training =====#
    parser.add_argument("-p", "--train_logp", type=bool, default=True)
    parser.add_argument("-r", "--train_mr", type=bool, default=True)
    parser.add_argument("-t", "--train_tpsa", type=bool, default=True)
    parser.add_argument("-mr", "--masking_rate", type=float, default=0.2)
    parser.add_argument("-R", "--radius", type=int, default=2)
    parser.add_argument("-er", "--erase_rate", type=float, default=0.8)
    parser.add_argument("-ml", "--max_len", type=int, default=50)


    parser.add_argument("-ep", "--epoch", type=int, default=100)
    parser.add_argument("-bs", "--batch_size", type=int, default=1024)
    parser.add_argument("-tbs", "--test_batch_size", type=int, default=1024)
    parser.add_argument("-nw", "--num_workers", type=int, default=12)

    # ===== Logging =====#
    parser.add_argument("-li", "--log_every", type=int, default= 5 * 10)  # Test: 10  #Default 40*10
    parser.add_argument("-vi", "--validate_every", type=int, default= 1000)  # Test:50 #Default 40*50
    parser.add_argument("-si", "--save_every", type=int, default= 1000)  # Test:50 #Default 40*100

    parser.add_argument("-mn", "--model_name", type=str, required=True)
    parser.add_argument("--log_path", type=str, default='runs')
    parser.add_argument("--ck_filename", type=str, default=None) #'model_4_256_xs_basic_000_000000300.tar')
    parser.add_argument("--dataset_path", type=str, default='./dataset/bal_xxs')

    args = parser.parse_args()#["-mn", "metric_test_0.5_masking"])

    # ===== Experiment Setup =====#
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.model_explain = make_model_comment(args)
    train_writer = SummaryWriter(join(args.log_path, args.model_name + '_train'))
    val_writer = SummaryWriter(join(args.log_path, args.model_name + '_val'))
    train_writer.add_text(tag='model', text_string='{}:{}'.format(args.model_name, args.model_explain),
                          global_step=0)
    logger = get_logger(join(args.log_path, args.model_name + '_train'))

    # ===== Loading Dataset =====#
    train_dataset_path = args.dataset_path + '/train'
    val_dataset_path = args.dataset_path + '/val'
    list_trains = get_dir_files(train_dataset_path)
    list_vals = get_dir_files(val_dataset_path)

    logger.info("##### Loading Train Dataloader #####")
    train_dataset = zincDataset(train_dataset_path, list_trains[0], args.num_workers)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  num_workers=args.num_workers,
                                  shuffle=True)

    logger.info("##### Loading Validation Dataloader #####")
    val_dataset = zincDataset(val_dataset_path, list_vals[0], args.num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.test_batch_size,
                                drop_last=True,
                                num_workers=args.num_workers,
                                shuffle=False)

    dataloader = {'train': train_dataloader, 'val': val_dataloader}
    logger.info("######## Starting Training ########")

    result = experiment(dataloader, args)

