from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import torch.nn.functional as F

##########################
# ===== Compute Loss =====#
##########################

# TODO: check loss and metric


def compute_loss(pred_x, ground_x, vocab_size):
    print(pred_x[0], ground_x[0])
    symbol_loss = F.cross_entropy(pred_x[:,:vocab_size], ground_x[:,0].detach().long())
    degree_loss = F.cross_entropy(pred_x[:,vocab_size:vocab_size+7], ground_x[:,1].detach().long())
    numH_loss = F.cross_entropy(pred_x[:,vocab_size+7:vocab_size+13], ground_x[:,2].detach().long())
    valence_loss = F.cross_entropy(pred_x[:,vocab_size+13:vocab_size+20], ground_x[:,3].detach().long())
    isarom_loss = F.cross_entropy(pred_x[:,vocab_size+20:vocab_size+23], ground_x[:,4].detach().long())
    partial_loss = F.mse_loss(pred_x[:,-1], ground_x[:,5])
    total_loss = symbol_loss + degree_loss + numH_loss + valence_loss + isarom_loss + partial_loss
    return symbol_loss, degree_loss, numH_loss, valence_loss, isarom_loss, partial_loss, total_loss


def compute_metric(pred_x, ground_x, vocab_size):
    symbol_acc = accuracy_score(ground_x[:, 0].detach().cpu().numpy(),
                                pred_x[:, :vocab_size].detach().max(dim=1)[1].cpu().numpy())
    degree_acc = accuracy_score(ground_x[:, 1].detach().cpu().numpy(),
                                pred_x[:, vocab_size:vocab_size+7].detach().max(dim=1)[1].cpu().numpy())
    numH_acc = accuracy_score(ground_x[:, 2].detach().cpu().numpy(),
                              pred_x[:, vocab_size+7:vocab_size+13].detach().max(dim=1)[1].cpu().numpy())
    valence_acc = accuracy_score(ground_x[:, 3].detach().cpu().numpy(),
                                 pred_x[:, vocab_size+13:vocab_size+20].detach().max(dim=1)[1].cpu().numpy())
    isarom_acc = accuracy_score(ground_x[:, 4].detach().cpu().numpy(),
                                pred_x[:, vocab_size+20:vocab_size+23].detach().max(dim=1)[1].cpu().numpy())
    partial_acc = mean_absolute_error(ground_x[:, 5].detach().cpu().numpy(),
                                      pred_x[:, -1].detach().cpu().numpy())
    return symbol_acc, degree_acc, numH_acc, valence_acc, isarom_acc, partial_acc


def compute_confusion(pred_x, ground_x, args):

    symbol_confusion = confusion_matrix(ground_x[:, 0].detach().cpu().numpy(),
                                        pred_x[:, :41].detach().max(dim=1)[1].cpu().numpy(),
                                        labels=range(args.vocab_size+1))
    degree_confusion = confusion_matrix(ground_x[:, 1].detach().cpu().numpy(),
                                        pred_x[:, 41:48].detach().max(dim=1)[1].cpu().numpy(),
                                        labels=range(args.degree_size+1))
    numH_confusion = confusion_matrix(ground_x[:, 2].detach().cpu().numpy(),
                                      pred_x[:, 48:54].detach().max(dim=1)[1].cpu().numpy(),
                                      labels=range(args.numH_size+1))
    valence_confusion = confusion_matrix(ground_x[:, 3].detach().cpu().numpy(),
                                         pred_x[:, 54:61].detach().max(dim=1)[1].cpu().numpy(),
                                         labels=range(args.valence_size+1))
    isarom_confusion = confusion_matrix(ground_x[:, 4].detach().cpu().numpy(),
                                        pred_x[:, 61:64].detach().max(dim=1)[1].cpu().numpy(),
                                        labels=range(args.isarom_size+1))

    return symbol_confusion, degree_confusion, numH_confusion, valence_confusion, isarom_confusion

