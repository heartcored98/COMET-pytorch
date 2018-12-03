import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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
# ===== Gconv, Readout, BN1D, ResBlock, Encoder =====#
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

    def forward(self, output_H):
        molvec = self.readout_fc(output_H)
        molvec = torch.mean(molvec, dim=1)
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

        if self.sc_type == 'no':  # no skip-connection
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
            gate_coef = self.sigmoid(x1 +x2)
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
            if i== 0:
                self.gconvs.append(
                    ResBlock(args.in_dim, self.out_dim, args.use_bn, args.use_attn, args.dp_rate, args.sc_type,
                             args.n_attn_heads))
            else:
                self.gconvs.append(
                    ResBlock(self.out_dim, self.out_dim, args.use_bn, args.use_attn, args.dp_rate, args.sc_type,
                             args.n_attn_heads))
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
        embed_x = self.embedding(x[:, :, 0])
        x = torch.cat((embed_x.float(), x[:, :, 1:].float()), 2)
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


####################################
# ===== Classifier & Regressor =====#
####################################


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, molvec_dim, vocab_size, dropout_rate=0):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.molvec_dim = molvec_dim
        self.vs = vocab_size

        self.fc1 = nn.Linear(self.molvec_dim + self.out_dim, self.in_dim)
        self.fc2 = nn.Linear(self.in_dim, self.in_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.param_initializer()

    def forward(self, X, molvec, idx_M):
        batch_size = X.shape[0]
        num_masking = idx_M.shape[1]

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
        self.reg_fc1 = nn.Linear(self.molvec_dim, self.molvec_dim // 2)
        self.reg_fc2 = nn.Linear(self.molvec_dim // 2, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, molvec):
        x = self.relu(self.reg_fc1(molvec))
        x = self.reg_fc2(x)
        return torch.squeeze(x)
