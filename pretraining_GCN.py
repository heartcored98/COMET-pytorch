import time
import argparse

from tensorboardX import SummaryWriter

from dataloader import *
from utils import *
from model import *


##########################
#===== Compute Loss =====#
##########################

def compute_loss(pred_x, ground_x, vocab_size):
    batch_size = ground_x.shape[0]
    num_masking = ground_x.shape[1]
    ground_x = ground_x.view(batch_size * num_masking, -1)
    
    symbol_loss = F.cross_entropy(pred_x[:,:vocab_size], ground_x[:, 0].detach())
    degree_loss = F.cross_entropy(pred_x[:,vocab_size:vocab_size+6], ground_x[:,1:7].detach().max(dim=1)[1])
    numH_loss = F.cross_entropy(pred_x[:,vocab_size+6:vocab_size+11], ground_x[:, 7:12].detach().max(dim=1)[1])
    valence_loss = F.cross_entropy(pred_x[:,vocab_size+11:vocab_size+17], ground_x[:,12:18].detach().max(dim=1)[1])
    isarom_loss = F.binary_cross_entropy(torch.sigmoid(pred_x[:,-1]), ground_x[:,-1].detach().float())
    # total_loss = symbol_loss + degree_loss + numH_loss + valence_loss + isarom_loss
    return symbol_loss, degree_loss, numH_loss, valence_loss, isarom_loss #total_loss


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
#===== Experiment =====#
########################

def train(models, optimizer, dataloader, epoch, cnt_iter, args):
    t = time.time()
    list_train_loss = list()
    epoch = epoch
    cnt_iter = cnt_iter
    reg_loss = nn.MSELoss()

    for epoch in range(epoch, args.epoch+1):
        for batch_idx, batch in enumerate(dataloader['train']):
            
            # Setting Train Mode
            for _, model in models.items():
                model.train()
            optimizer.zero_grad()

            # Get Batch Sample from DataLoader
            input_X, A, mol_prop, ground_X, idx_M = batch
            input_X = Variable(torch.from_numpy(input_X)).to(args.device).long()
            A = Variable(torch.from_numpy(A)).to(args.device).float()
            mol_prop = Variable(torch.from_numpy(mol_prop)).to(args.device).float()
            logp, mr, tpsa = mol_prop[:,0], mol_prop[:,1], mol_prop[:,2]
            ground_X = Variable(torch.from_numpy(ground_X)).to(args.device).long()
            idx_M = Variable(torch.from_numpy(idx_M)).to(args.device).long()

            # Encoding Molecule
            X, A, molvec = models['encoder'](input_X, A)
            pred_mask = models['classifier'](X, molvec, idx_M)

            # Compute Mask Task Loss & Property Regression Loss
            symbol_loss, degree_loss, numH_loss, valence_loss, isarom_loss = compute_loss(pred_mask, ground_X, args.vocab_size)
            mask_loss = symbol_loss + degree_loss + numH_loss + valence_loss + isarom_loss
            loss = mask_loss

            if args.train_logp:
                pred_logp = models['logP'](molvec)
                logP_loss = reg_loss(pred_logp, logp)
                loss += logP_loss
                train_writer.add_scalar('3.auxilary/logP', logP_loss, cnt_iter)
            if args.train_mr:
                pred_mr = models['mr'](molvec)
                mr_loss = reg_loss(pred_mr, mr)
                loss += mr_loss
                train_writer.add_scalar('3.auxilary/mr', mr_loss, cnt_iter)

            if args.train_tpsa:
                pred_tpsa = models['tpsa'](molvec)
                tpsa_loss = reg_loss(pred_tpsa, tpsa)
                loss += tpsa_loss
                train_writer.add_scalar('3.auxilary/tpsa', tpsa_loss, cnt_iter)

            train_writer.add_scalar('2.mask/symbol', symbol_loss, cnt_iter)
            train_writer.add_scalar('2.mask/degree', degree_loss, cnt_iter)
            train_writer.add_scalar('2.mask/numH', numH_loss, cnt_iter)
            train_writer.add_scalar('2.mask/valence', valence_loss, cnt_iter)
            train_writer.add_scalar('2.mask/isarom', isarom_loss, cnt_iter)

            train_writer.add_scalar('1.status/total', loss, cnt_iter)
            train_writer.add_scalar('1.status/mask', mask_loss, cnt_iter)

            # Backprogating and Updating Parameter
            loss.backward()
            optimizer.step(cnt_iter)
            train_writer.add_scalar('1.status/lr', optimizer.rate(cnt_iter), cnt_iter)
            cnt_iter += 1
            setattr(args, 'epoch_now', epoch)
            setattr(args, 'iter_now', cnt_iter)

            # Prompting Status
            if cnt_iter % args.log_every == 0:
                output = "[TRAIN] E:{:3}. P:{:>2.1f}%. Loss:{:>9.3}. Mask Loss:{:>9.3}. {:4.1f} mol/sec. Iter:{:6}.  Elapsed:{:6.1f} sec."
                elapsed = time.time() - t
                process_speed = (args.batch_size * args.log_every) / elapsed
                output = output.format(epoch, batch_idx / len(dataloader['train']) * 100.0, loss, mask_loss, process_speed, cnt_iter, elapsed,)
                t = time.time()
                logger.info(output)

            # Validate Model
            if cnt_iter % args.validate_every == 0:
                optimizer.zero_grad()
                validate(models, dataloader['val'], args, cnt_iter=cnt_iter, epoch=epoch)
                t = time.time()

            # Save Model
            if cnt_iter % args.save_every == 0:
                filename = save_checkpoint(epoch, cnt_iter, models, optimizer, args)
                logger.info('Saved Model as {}'.format(filename))

            del batch
                
    logger.info('Training Completed')


def validate(models, data_loader, args, **kwargs):

    t = time.time()
    cnt_iter = kwargs['cnt_iter']
    epoch = kwargs['epoch']
    temp_iter = 0
    reg_loss = nn.MSELoss()

    list_mask_loss = []
    list_symbol_loss = []
    list_degree_loss = []
    list_numH_loss = []
    list_valence_loss = []
    list_isarom_loss = []
    list_logP_loss = []
    list_mr_loss = []
    list_tpsa_loss = []

    # Initialization Model with Evaluation Mode
    for _, model in models.items():
        model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_X, A, mol_prop, ground_X, idx_M = batch
            input_X = Variable(torch.from_numpy(input_X)).to(args.device).long()
            A = Variable(torch.from_numpy(A)).to(args.device).float()
            mol_prop = Variable(torch.from_numpy(mol_prop)).to(args.device).float()
            logp, mr, tpsa = mol_prop[:,0], mol_prop[:,1], mol_prop[:,2]
            ground_X = Variable(torch.from_numpy(ground_X)).to(args.device).long()
            idx_M = Variable(torch.from_numpy(idx_M)).to(args.device).long()

            # Encoding Molecule
            X, A, molvec = models['encoder'](input_X, A)
            pred_mask = models['classifier'](X, molvec, idx_M)
            
            # Compute Mask Task Loss & Property Regression Loss
            symbol_loss, degree_loss, numH_loss, valence_loss, isarom_loss = compute_loss(pred_mask, ground_X, args.vocab_size)
            list_symbol_loss.append(symbol_loss.item())
            list_degree_loss.append(degree_loss.item())
            list_numH_loss.append(numH_loss.item())
            list_valence_loss.append(valence_loss.item())
            list_isarom_loss.append(isarom_loss.item())
            list_mask_loss.append((symbol_loss + degree_loss + numH_loss + valence_loss + isarom_loss).item())

            if args.train_logp:
                pred_logp = models['logP'](molvec)
                list_logP_loss.append(reg_loss(pred_logp, logp).item())
            if args.train_mr:
                pred_mr = models['mr'](molvec)
                list_mr_loss.append(reg_loss(pred_mr, mr).item())
            if args.train_tpsa:
                pred_tpsa = models['tpsa'](molvec)
                list_tpsa_loss.append(reg_loss(pred_tpsa, tpsa).item())

            temp_iter += 1   

            # Prompting Status
            if temp_iter % (args.log_every * 10) == 0:
                output = "[VALID] E:{:3}. P:{:>2.1f}%. {:4.1f} mol/sec. Iter:{:6}.  Elapsed:{:6.1f} sec."
                elapsed = time.time() - t
                process_speed = (args.test_batch_size * args.log_every) / elapsed
                output = output.format(epoch, batch_idx / len(data_loader) * 100.0, process_speed, temp_iter, elapsed,)
                t = time.time()
                logger.info(output)

            del batch
                
    mask_loss = np.mean(np.array(list_mask_loss))
    symbol_loss = np.mean(np.array(list_symbol_loss))
    degree_loss = np.mean(np.array(list_degree_loss))
    numH_loss = np.mean(np.array(list_numH_loss))
    valence_loss = np.mean(np.array(list_valence_loss))
    isarom_loss = np.mean(np.array(list_isarom_loss))

    loss = mask_loss
    if args.train_logp:
        logP_loss = np.mean(np.array(list_logP_loss))
        loss += logP_loss
        val_writer.add_scalar('3.auxilary/logP', logP_loss, cnt_iter)
    if args.train_mr:
        mr_loss = np.mean(np.array(list_mr_loss))
        loss += mr_loss
        val_writer.add_scalar('3.auxilary/mr', mr_loss, cnt_iter)

    if args.train_tpsa:
        tpsa_loss = np.mean(np.array(list_tpsa_loss))
        loss += tpsa_loss
        val_writer.add_scalar('3.auxilary/tpsa', tpsa_loss, cnt_iter)

    val_writer.add_scalar('2.mask/symbol', symbol_loss, cnt_iter)
    val_writer.add_scalar('2.mask/degree', degree_loss, cnt_iter)
    val_writer.add_scalar('2.mask/numH', numH_loss, cnt_iter)
    val_writer.add_scalar('2.mask/valence', valence_loss, cnt_iter)
    val_writer.add_scalar('2.mask/isarom', isarom_loss, cnt_iter)

    val_writer.add_scalar('1.status/total', loss, cnt_iter)
    val_writer.add_scalar('1.status/mask', mask_loss, cnt_iter)

    # Log model weight historgram
    log_histogram(models, val_writer, cnt_iter)
    
    """
    # Calculate overall MAE and STD value      
    if args.train_logp:
        logp_mae = mean_absolute_error(list_logp, list_pred_logp)
        logp_std = np.std(np.array(list_logp)-np.array(list_pred_logp))
        
    if args.train_mr:
        mr_mae = mean_absolute_error(list_mr, list_pred_mr)
        mr_std = np.std(np.array(list_mr)-np.array(list_pred_mr))
        
    if args.train_tpsa:
        tpsa_mae = mean_absolute_error(list_tpsa, list_pred_tpsa)
        tpsa_std = np.std(np.array(list_tpsa)-np.array(list_pred_tpsa))
    """

    output = "[V] E:{:3}. P:{:>2.1f}%. Loss:{:>9.3}. Mask Loss:{:>9.3}. {:4.1f} mol/sec. Iter:{:6}.  Elapsed:{:6.1f} sec."
    elapsed = time.time() - t
    process_speed = (args.test_batch_size * args.log_every) / elapsed
    output = output.format(epoch, batch_idx / len(data_loader) * 100.0, loss, mask_loss, process_speed, cnt_iter, elapsed,)
    t = time.time()
    logger.info(output)

    torch.cuda.empty_cache()

def experiment(dataloader, args):
    ts = time.time()
    
    # Construct Model
    encoder = Encoder(args)
    classifier = Classifier(args.in_dim, args.out_dim, args.molvec_dim, args.vocab_size, args.dp_rate, ACT2FN[args.act])
    models = {'encoder': encoder, 'classifier': classifier}
    if args.train_logp:
        models.update({'logP': Regressor(args.molvec_dim, args.dp_rate, ACT2FN[args.act])})
    if args.train_mr:
        models.update({'mr': Regressor(args.molvec_dim, args.dp_rate, ACT2FN[args.act])})
    if args.train_tpsa:
        models.update({'tpsa': Regressor(args.molvec_dim, args.dp_rate, ACT2FN[args.act])})
        
    # Initialize Optimizer
    logger.info('####### Model Constructed #######')
    trainable_parameters = list()
    for key, model in models.items():
        model.to(args.device)
        trainable_parameters += list(filter(lambda p: p.requires_grad, model.parameters()))
        logger.info('{:10}: {:>10} parameters'.format(key, sum(p.numel() for p in model.parameters())))
        setattr(args, '{}_param'.format(key), sum(p.numel() for p in model.parameters()))
    logger.info('#################################')
    
    if args.optim == 'ADAM':
        optimizer = optim.Adam(trainable_parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
    elif args.optim == 'RMSProp':
        optimizer = optim.RMSprop(trainable_parameters, lr=0)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(trainable_parameters, lr=0)
    else:
        assert False, "Undefined Optimizer Type"

    # Reload Checkpoint Model
    epoch = 0
    cnt_iter = 0
    if args.ck_filename:
        epoch, cnt_iter, models, optimizer = load_checkpoint(models, optimizer, args.ck_filename, args)
        logger.info('Loaded Model from {}'.format(args.ck_filename))
    
    optimizer = NoamOpt(args.out_dim, args.lr_factor, args.lr_step, optimizer)
    log_histogram(models, val_writer, cnt_iter)

    # Train Model
    train(models, optimizer, dataloader, epoch, cnt_iter, args)

    # Logging Experiment Result
    te = time.time()    
    args.elapsed = te-ts
    logger.info('Training Completed')



if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description='Add logP, TPSA, MR, PBF value on .smi files')
    #===== Model Definition =====#
    parser.add_argument("-v", "--vocab_size", type=int, default=41)
    parser.add_argument("-i", "--in_dim", type=int, default=59)
    parser.add_argument("-o", "--out_dim", type=int, default=256)
    parser.add_argument("-m", "--molvec_dim", type=int, default=256)

    parser.add_argument("-n", "--n_layer", type=int, default=4)
    parser.add_argument("-k", "--n_attn_heads", type=int, default=8)
    parser.add_argument("-c", "--sc_type", type=str, default='sc')

    parser.add_argument("-a", "--use_attn", type=bool, default=True)
    parser.add_argument("-b", "--use_bn", type=bool, default=True)
    parser.add_argument("-e", "--emb_train", type=bool, default=True)
    parser.add_argument("-dp", "--dp_rate", type=float, default=0.1)

    #===== Optimizer =====#
    parser.add_argument("-u", "--optim", type=str, default='ADAM')
    parser.add_argument("-lf", "--lr_factor", type=float, default=2.0)
    parser.add_argument("-ls", "--lr_step", type=int, default=4000)

    #===== Training =====#
    parser.add_argument("-p", "--train_logp", type=bool, default=True)
    parser.add_argument("-r", "--train_mr", type=bool, default=True)
    parser.add_argument("-t", "--train_tpsa", type=bool, default=True)

    parser.add_argument("-ep", "--epoch", type=int, default=100)
    parser.add_argument("-bs", "--batch_size", type=int, default=512)
    parser.add_argument("-tbs", "--test_batch_size", type=int, default=512)

    #===== Logging =====#
    parser.add_argument("-li", "--log_every", type=int, default=10*10) #Test: 10  #Default 40*10
    parser.add_argument("-vi", "--validate_every", type=int, default=50*40) #Test:50 #Default 40*50
    parser.add_argument("-si", "--save_every", type=int, default=40*100) #Test:50 #Default 40*100

    parser.add_argument("-mn", "--model_name", type=str, required=True)
    parser.add_argument("--log_path", type=str, default='runs')
    parser.add_argument("--ck_filename", type=str, default=None) #'model_ck_000_000000100.tar')
    parser.add_argument("--dataset_path", type=str, default='./dataset/data_s')

    args = parser.parse_args()

    #===== Experiment Setup =====#
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.model_explain = make_model_comment(args)
    train_writer = SummaryWriter(join(args.log_path, args.model_name+'_train'))
    val_writer = SummaryWriter(join(args.log_path, args.model_name+'_val'))
    train_writer.add_text(tag='model', text_string='{}:{}'.format(args.model_name, args.model_explain), global_step=0)
    logger = get_logger(join(args.log_path, args.model_name+'_train'))

    #===== Loading Dataset =====#
    train_dataset_path = args.dataset_path + '/train'
    val_dataset_path = args.dataset_path + '/val'
    list_trains = get_dir_files(train_dataset_path)
    list_vals = get_dir_files(val_dataset_path)

    train_dataloader = zincDataLoader(join(train_dataset_path, list_trains[0]),
                                      batch_size=args.batch_size,
                                      drop_last=False,
                                      shuffle_batch=True,
                                      num_workers=8)

    val_dataloader = zincDataLoader(join(val_dataset_path, list_vals[0]),
                                      batch_size=args.test_batch_size,
                                      drop_last=False,
                                      shuffle_batch=False,
                                      num_workers=8)

    dataloader = {'train': train_dataloader, 'val': val_dataloader}
    result = experiment(dataloader, args)

