import torch
import helper
import torch.distributed as dist

def compute_embedding(backbone, subset_data_loader, full_dataloader):
    """Compute CLS embedding and prepare for Tensorboard"""
    device = next(backbone.parameters()).device

    embs_list = []
    imgs_list = []
    labels = []

    for img, y in subset_data_loader:
        img = img.to(device)
        embs_list.append(backbone(img).detach().cpu())
        imgs_list.append(img.cpu())
        labels.extend([full_dataloader.dataset.classes[i] for i in y.tolist()])

    embs = torch.cat(embs_list, dim=0)
    imgs = torch.cat(imgs_list, dim=0)
    return embs, imgs, labels
