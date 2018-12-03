


import time
import argparse
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter


from dataloader import *
from utils import *



class Attention(nn.Module):
    def __init__(self, input_dim, output_dim, num_attn_head, dropout=0.1):
        super(Attention, self).__init__()   

        self.num_attn_heads = num_attn_head
        self.attn_dim = output_dim // num_attn_head
        self.projection = nn.ModuleList([nn.Linear(input_dim, self.attn_dim) for i in range(self.num_attn_heads)])
        self.coef_matrix = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.attn_dim, self.attn_dim)) for i in range(self.num_attn_heads)])
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.param_initializer()

    def forward(self, X, A):
        list_X_head = list()
        for i in range(self.num_attn_heads):
            X_projected = self.projection[i](X)
            attn_matrix = self.attn_coeff(X_projected, A, self.coef_matrix[i])
            X_head = torch.matmul(attn_matrix, X_projected)
            list_X_head.append(X_head)
            
        X = torch.cat(list_X_head, dim=2)
        X = self.relu(X)
        return X
            
    def attn_coeff(self, X_projected, A, C):
        X = torch.einsum('akj,ij->aki', (X_projected, C))
        attn_matrix = torch.matmul(X, torch.transpose(X_projected, 1, 2)) 
        attn_matrix = torch.mul(A, attn_matrix)
        attn_matrix = self.dropout(self.tanh(attn_matrix))
        return attn_matrix
    
    def param_initializer(self):
        for i in range(self.num_attn_heads):    
            nn.init.xavier_normal_(self.projection[i].weight.data)
            nn.init.xavier_normal_(self.coef_matrix[i].data)
            

#####################################################
#===== Gconv, Readout, BN1D, ResBlock, Encoder =====#
#####################################################

class GConv(nn.Module):
    def __init__(self, input_dim, output_dim, attn):
        super(GConv, self).__init__()
        self.attn = attn
        if self.attn is None:
            self.fc = nn.Linear(input_dim, output_dim)
            nn.init.xavier_normal_(self.fc.weight.data)
        
    def forward(self, X, A):
        if self.attn is None:
            x = self.fc(X)
            x = torch.matmul(A, x)
        else:
            x = self.attn(X, A)            
        return x, A
    
    
class Readout(nn.Module):
    def __init__(self, out_dim, molvec_dim):
        super(Readout, self).__init__()
        self.readout_fc = nn.Linear(out_dim, molvec_dim)
        nn.init.xavier_normal_(self.readout_fc.weight.data)
        self.relu = nn.ReLU()
        
    def forward(self, output_H):
        molvec = self.readout_fc(output_H)
        molvec = self.relu(torch.sum(molvec, dim=1))
        return molvec


class BN1d(nn.Module):
    def __init__(self, out_dim, use_bn):
        super(BN1d, self).__init__()
        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(out_dim)
             
    def forward(self, x): 
        if not self.use_bn:
            return  x
        origin_shape = x.shape
        x = x.view(-1, origin_shape[-1])
        x = self.bn(x)
        x = x.view(origin_shape)
        return x
    
    
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn, use_attn, dp_rate, sc_type, n_attn_head=None):
        super(ResBlock, self).__init__()   
        self.use_bn = use_bn
        self.sc_type = sc_type
        
        attn = Attention(in_dim, out_dim, n_attn_head) if use_attn else None
        self.gconv = GConv(in_dim, out_dim, attn)
        
        self.bn1 = BN1d(out_dim, use_bn)
        self.dropout = nn.Dropout2d(p=dp_rate)
        self.relu = nn.ReLU()
        
        if not self.sc_type in ['no', 'gsc', 'sc']:
            raise Exception

        if self.sc_type != 'no':
            self.bn2 = BN1d(out_dim, use_bn)
            self.shortcut = nn.Sequential()
            if in_dim != out_dim:
                self.shortcut.add_module('shortcut', nn.Linear(in_dim, out_dim, bias=False))
                
        if self.sc_type == 'gsc':
            self.g_fc1 = nn.Linear(out_dim, out_dim, bias=True)
            self.g_fc2 = nn.Linear(out_dim, out_dim, bias=True)
            self.sigmoid = nn.Sigmoid()

    def forward(self, X, A):     
        x, A = self.gconv(X, A)

        if self.sc_type == 'no': #no skip-connection
            x = self.relu(self.bn1(x))
            return self.dropout(x), A
        
        elif self.sc_type == 'sc': # basic skip-connection
            x = self.relu(self.bn1(x))
            x = x + self.shortcut(X)          
            return self.dropout(self.relu(self.bn2(x))), A
        
        elif self.sc_type == 'gsc': # gated skip-connection
            x = self.relu(self.bn1(x)) 
            x1 = self.g_fc1(self.shortcut(X))
            x2 = self.g_fc2(x)
            gate_coef = self.sigmoid(x1+x2)
            x = torch.mul(x1, gate_coef) + torch.mul(x2, 1.0-gate_coef)
            return self.dropout(self.relu(self.bn2(x))), A

        
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.bs = args.batch_size
        self.molvec_dim = args.molvec_dim
        self.embedding = self.create_emb_layer(args.vocab_size, args.emb_train) 
        self.out_dim = args.out_dim
        
        self.gconvs = nn.ModuleList()
        for i in range(args.n_layer):
            if i==0:
                self.gconvs.append(ResBlock(args.in_dim, self.out_dim, args.use_bn, args.use_attn, args.dp_rate, args.sc_type, args.n_attn_heads))
            else:
                self.gconvs.append(ResBlock(self.out_dim, self.out_dim, args.use_bn, args.use_attn, args.dp_rate, args.sc_type, args.n_attn_heads))
        self.readout = Readout(self.out_dim, self.molvec_dim)
    
    def forward(self, input_X, A):   
        x, A, molvec = self.encoder(input_X, A)
        return x, A, molvec
     
    def encoder(self, input_X, A):
        x = self._embed(input_X)
        for i, module in enumerate(self.gconvs):
            x, A = module(x, A)
        molvec = self.readout(x)
        return x, A, molvec
    
    def _embed(self, x):
        embed_x = self.embedding(x[:,:,0])
        x = torch.cat((embed_x.float(), x[:,:,1:].float()), 2)
        return x 

    def create_emb_layer(self, vocab_size, emb_train=False):
        emb_layer = nn.Embedding(vocab_size, vocab_size)
        weight_matrix = torch.zeros((vocab_size, vocab_size))
        for i in range(vocab_size):
            weight_matrix[i][i] = 1
        emb_layer.load_state_dict({'weight': weight_matrix})

        if not emb_train:
            emb_layer.weight.requires_grad = False
        return emb_layer


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
    total_loss = symbol_loss + degree_loss + numH_loss + valence_loss + isarom_loss
    return total_loss
    

####################################
#===== Classifier & Regressor =====#
####################################


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, molvec_dim, vocab_size, dropout_rate=0):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.molvec_dim = molvec_dim
        self.vs = vocab_size
    
        self.fc1 = nn.Linear(self.molvec_dim + self.out_dim, args.in_dim)
        self.fc2 = nn.Linear(self.in_dim, args.in_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.param_initializer()
        
    def forward(self, X, molvec, idx_M):
        batch_size = X.shape[0]
        #print('idx_M', idx_M.shape)
        num_masking = idx_M.shape[1]
        probs_atom = list()
        probs_degree = list()
        probs_numH = list()
        probs_valence = list()
        probs_isarom = list()
        
        molvec = torch.unsqueeze(molvec, 1)
        molvec = molvec.expand(batch_size, num_masking, molvec.shape[-1])
        
        list_concat_x = list()
        for i in range(batch_size):
            target_x = torch.index_select(X[i], 0, idx_M[i])
            concat_x = torch.cat((target_x, molvec[i]), dim=1)
            list_concat_x.append(concat_x)
            
        concat_x = torch.stack(list_concat_x)
        pred_x = self.classify(concat_x)
        pred_x = pred_x.view(batch_size * num_masking, -1)
        return pred_x
    
    def classify(self, concat_x):
        x = self.relu(self.fc1(concat_x))
        x = self.fc2(x)
        return x
    
    def param_initializer(self):
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
    

class Regressor(nn.Module):
    def __init__(self, molvec_dim, dropout_rate):
        super(Regressor, self).__init__()

        self.molvec_dim = molvec_dim
        self.reg_fc1 = nn.Linear(self.molvec_dim, self.molvec_dim//2)
        self.reg_fc2 = nn.Linear(self.molvec_dim//2, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, molvec):
        x = self.relu(self.reg_fc1(molvec))
        x = self.reg_fc2(x)
        return torch.squeeze(x)  


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
        rate = self.rate(step+1)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
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
        epoch_train_loss = 0
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
            mask_loss = compute_loss(pred_mask, ground_X, args.vocab_size)
            loss = mask_loss

            if args.train_logp:
                pred_logp = models['logP'](molvec)
                logP_loss = reg_loss(pred_logp, logp)
                loss += logP_loss
                train_writer.add_scalar('auxilary/logP', logP_loss, cnt_iter)
            if args.train_mr:
                pred_mr = models['mr'](molvec)
                mr_loss = reg_loss(pred_mr, mr)
                loss += mr_loss
                train_writer.add_scalar('auxilary/mr', mr_loss, cnt_iter)

            if args.train_tpsa:
                pred_tpsa = models['tpsa'](molvec)
                tpsa_loss = reg_loss(pred_tpsa, tpsa)
                loss += tpsa_loss
                train_writer.add_scalar('auxilary/tpsa', tpsa_loss, cnt_iter)

            train_writer.add_scalar('loss/total', loss, cnt_iter)
            train_writer.add_scalar('loss/mask', mask_loss, cnt_iter)


            epoch_train_loss += loss / len(batch)
            list_train_loss.append({'epoch':batch_idx/len(dataloader['train'])+epoch, 'train_loss':loss})
            
            # Backprogating and Updating Parameter
            loss.backward()
            optimizer.step(cnt_iter)
            cnt_iter += 1   

            # Save Model 
            if cnt_iter % args.save_every == 0:
                filename = save_checkpoint(epoch, cnt_iter, models, optimizer, args)
                logger.info('Saved Model as {}'.format(filename))

            # Validate Model
            if cnt_iter % args.validate_every == 0:
                optimizer.zero_grad()
                validate(models, dataloader['val'], args, cnt_iter=cnt_iter, epoch=epoch)
                t = time.time()

            # Prompting Status
            if cnt_iter % args.log_every == 0:
                output = "[TRAIN] E:{:3}. P:{:>2.1f}%. Loss:{:>9.3}. Mask Loss:{:>9.3}. {:4.1f} mol/sec. Iter:{:6}.  Elapsed:{:6.1f} sec."
                elapsed = time.time() - t
                process_speed = (args.batch_size * args.log_every) / elapsed
                output = output.format(epoch, batch_idx / len(dataloader['train']) * 100.0, loss, mask_loss, process_speed, cnt_iter, elapsed,)
                t = time.time()
                logger.info(output)
                
    logger.info('Training Completed')


def validate(models, data_loader, args, **kwargs):

    t = time.time()
    epoch_val_loss = 0
    cnt_iter = kwargs['cnt_iter']
    epoch = kwargs['epoch']
    temp_iter = 0
    reg_loss = nn.MSELoss()
    
    mask_loss = []
    logP_loss = []
    mr_loss = []
    tpsa_loss = []
    
    list_logp, list_pred_logp = [], []
    list_mr, list_pred_mr = [], []
    list_tpsa, list_pred_tpsa = [], []
    logp_mae, logp_std, mr_mae, mr_std, tpsa_mae, tpsa_std = 0, 0, 0, 0, 0, 0

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
            mask_loss.append(compute_loss(pred_mask, ground_X, args.vocab_size).item())

            if args.train_logp:
                pred_logp = models['logP'](molvec)
                logP_loss.append(reg_loss(pred_logp, logp).item())
            if args.train_mr:
                pred_mr = models['mr'](molvec)
                mr_loss.append(reg_loss(pred_mr, mr).item())
            if args.train_tpsa:
                pred_tpsa = models['tpsa'](molvec)
                tpsa_loss.append(reg_loss(pred_tpsa, tpsa).item())

            temp_iter += 1   

            # Prompting Status
            if temp_iter % (args.log_every * 4) == 0:
                output = "[VALID] E:{:3}. P:{:>2.1f}%. {:4.1f} mol/sec. Iter:{:6}.  Elapsed:{:6.1f} sec."
                elapsed = time.time() - t
                process_speed = (args.test_batch_size * args.log_every) / elapsed
                output = output.format(epoch, batch_idx / len(data_loader) * 100.0, process_speed, temp_iter, elapsed,)
                t = time.time()
                logger.info(output)
                
    mask_loss = np.mean(np.array(mask_loss))
    loss = mask_loss
    if args.train_logp:
        logP_loss = np.mean(np.array(logP_loss))
        loss += logP_loss
        val_writer.add_scalar('auxilary/logP', logP_loss, cnt_iter)
    if args.train_mr:
        mr_loss = np.mean(np.array(mr_loss))
        loss += mr_loss
        val_writer.add_scalar('auxilary/mr', mr_loss, cnt_iter)

    if args.train_tpsa:
        tpsa_loss = np.mean(np.array(tpsa_loss))
        loss += tpsa_loss
        val_writer.add_scalar('auxilary/tpsa', tpsa_loss, cnt_iter)

    
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
        
    val_writer.add_scalar('loss/total', loss, cnt_iter)
    val_writer.add_scalar('loss/mask', mask_loss, cnt_iter)

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
    classifier = Classifier(args.in_dim, args.out_dim, args.molvec_dim, args.vocab_size, args.dp_rate)
    models = {'encoder': encoder, 'classifier': classifier}
    if args.train_logp:
        models.update({'logP': Regressor(args.molvec_dim, args.dp_rate)})
    if args.train_mr:
        models.update({'mr': Regressor(args.molvec_dim, args.dp_rate)})
    if args.train_tpsa:
        models.update({'tpsa': Regressor(args.molvec_dim, args.dp_rate)})
        
    # Initialize Optimizer
    logger.info('####### Model Constructed #######')
    trainable_parameters = list()
    for key, model in models.items():
        model.to(args.device)
        trainable_parameters += list(filter(lambda p: p.requires_grad, model.parameters()))
        logger.info('{:10}: {:>10} parameters'.format(key, sum(p.numel() for p in model.parameters() if p.requires_grad)))
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

    """
    # Initialize Data Logger
    list_train_loss = list()
    list_val_loss = list()
    list_logp_mae = list()
    list_logp_std = list()
    list_mr_mae = list()
    list_mr_std = list()
    list_tpsa_mae = list()
    list_tpsa_std = list()
    """

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
    parser.add_argument("-m", "--molvec_dim", type=int, default=512)

    parser.add_argument("-n", "--n_layer", type=int, default=6)
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
    parser.add_argument("-li", "--log_every", type=int, default=10) #Test: 10  #Default 40*10
    parser.add_argument("-vi", "--validate_every", type=int, default=50) #Test:50 #Default 40*50
    parser.add_argument("-si", "--save_every", type=int, default=50) #Test:50 #Default 40*100

    parser.add_argument("-mn", "--model_name", type=str, required=True)
    parser.add_argument("--log_path", type=str, default='runs')
    parser.add_argument("--ck_filename", type=str, default=None) #'model_ck_000_000000200.tar'
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

