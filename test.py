symbol_acc = accuracy_score(pred_x[:, :vocab_size].detach().max(dim=1)[1].cpu().numpy(),
                            ground_x[:, 0].detach().cpu().numpy())
degree_acc = accuracy_score(pred_x[:, vocab_size:vocab_size + 6].detach().max(dim=1)[1].cpu().numpy(),
                            ground_x[:, 1:7].detach().max(dim=1)[1].cpu().numpy())
numH_acc = accuracy_score(pred_x[:, vocab_size + 6:vocab_size + 11].detach().max(dim=1)[1].cpu().numpy(),
                          ground_x[:, 7:12].detach().max(dim=1)[1].cpu().numpy())
valence_acc = accuracy_score(pred_x[:, vocab_size + 11:vocab_size + 17].detach().max(dim=1)[1].cpu().numpy(),
                             ground_x[:, 12:18].detach().max(dim=1)[1].cpu().numpy())
isarom_acc = accuracy_score((torch.sigmoid(pred_x[:, -1]) + 1).floor_().cpu().numpy(),
                            ground_x[:, -1].detach().cpu().numpy())