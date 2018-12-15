from sklearn.metrics import accuracy_score, confusion_matrix
import torch.nn.functional as F

##########################
# ===== Compute Loss =====#
##########################

def compute_loss(pred_x, ground_x):
    """
    batch_size = ground_x.shape[0]
    num_masking = ground_x.shape[1]
    ground_x = ground_x.view(batch_size * num_masking, -1)
    """

    # ground_x -= 1 #remove [mask] token which occupied index 0
    # ground_x[ground_x < 0] = 0
    symbol_loss = F.cross_entropy(pred_x[:, :41], ground_x[:, 0].detach())

    degree_loss = F.cross_entropy(pred_x[:, 41:48], ground_x[:, 1].detach())
    numH_loss = F.cross_entropy(pred_x[:, 48:54], ground_x[:, 2].detach())
    valence_loss = F.cross_entropy(pred_x[:, 54:61], ground_x[:, 3].detach())
    isarom_loss = F.cross_entropy(pred_x[:, 61:64], ground_x[:, 4].detach())
    return symbol_loss, degree_loss, numH_loss, valence_loss, isarom_loss


def compute_metric(pred_x, ground_x):

    symbol_acc = accuracy_score(ground_x[:, 0].detach().cpu().numpy(),
                                pred_x[:, :41].detach().max(dim=1)[1].cpu().numpy())
    degree_acc = accuracy_score(ground_x[:, 1].detach().cpu().numpy(),
                                pred_x[:, 41:48].detach().max(dim=1)[1].cpu().numpy())
    numH_acc = accuracy_score(ground_x[:, 2].detach().cpu().numpy(),
                              pred_x[:, 48:54].detach().max(dim=1)[1].cpu().numpy())
    valence_acc = accuracy_score(ground_x[:, 3].detach().cpu().numpy(),
                                 pred_x[:, 54:61].detach().max(dim=1)[1].cpu().numpy())
    isarom_acc = accuracy_score(ground_x[:, 4].detach().cpu().numpy(),
                                pred_x[:, 61:64].detach().max(dim=1)[1].cpu().numpy())
    return symbol_acc, degree_acc, numH_acc, valence_acc, isarom_acc


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

