import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import helper


def compute_embedding(backbone, dataloader, label_mapping, return_subset=False, subset_size=0):
    """Compute CLS embedding and prepare for TensorBoard.
     Parameters
     ----------
     backbone : model
     dataloader : torch.utils.data.DataLoader
         Validation dataloader that does not apply any augmentations. Just
         casting to tensor and then normalizing.
     Returns
     -------
     embs : torch.Tensor
         Embeddings of shape `(n_samples, out_dim)`.
     imgs : torch.Tensor
         Images of shape `(n_samples, 3, height, width)`.
     labels : list
         List of strings representing the classes.
     """
    backbone.eval()
    embeddings = None
    images = None
    labels = None

    for img, lab, index in dataloader:
        img = img.cuda(non_blocking=True)
        img = (img * 0.224) + 0.45 # undo norm
        lab = lab.cuda(non_blocking=True).view(-1,1).float()
        index = index.cuda(non_blocking=True)
        embs = backbone(img).clone()

        # Initialise storage matrix
        if dist.get_rank() == 0 and embeddings is None:
            embeddings = torch.zeros(len(dataloader.dataset), embs.shape[-1])
            embeddings = embeddings.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {embeddings.shape}")

        if dist.get_rank() == 0 and images is None:
            images = torch.zeros(len(dataloader.dataset), img.shape[1], img.shape[2], img.shape[3])
            images = images.cuda(non_blocking=True)
            print(f"Storing images into tensor of shape {images.shape}")

        if dist.get_rank() == 0 and labels is None:
            labels = torch.zeros(len(dataloader.dataset), lab.shape[-1])
            labels = labels.cuda(non_blocking=True)
            print(f"Storing labels into tensor of shape {labels.shape}")

        # Share features between processes
        embs_all = torch.empty(
            dist.get_world_size(),
            embs.size(0),
            embs.size(1),
            dtype=embs.dtype,
            device=embs.device,
        )
        embs_l = list(embs_all.unbind(0))
        embs_all_reduce = torch.distributed.all_gather(embs_l, embs, async_op=True)
        embs_all_reduce.wait()

        # Share images between processes
        img_all = torch.empty(
            dist.get_world_size(),
            img.size(0),
            img.size(1),
            img.size(2),
            img.size(3),
            dtype=img.dtype,
            device=img.device,
        )
        img_l = list(img_all.unbind(0))
        img_all_reduce = torch.distributed.all_gather(img_l, img, async_op=True)
        img_all_reduce.wait()

        # Share labels between processes
        lab_all = torch.empty(
            dist.get_world_size(),
            lab.size(0),
            lab.size(1),
            dtype=lab.dtype,
            device=lab.device,
        )
        lab_l = list(lab_all.unbind(0))
        lab_all_reduce = torch.distributed.all_gather(lab_l, lab, async_op=True)
        lab_all_reduce.wait()

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # update storage feature matrix
        if dist.get_rank() == 0:
            embeddings.index_copy_(0, index_all, torch.cat(embs_l))
            images.index_copy_(0, index_all, torch.cat(img_l))
            labels.index_copy_(0, index_all, torch.cat(lab_l))
            
    if dist.get_rank() == 0:
        labels = [dataloader.dataset.classes[int(i)] for i in labels.view(1,-1)[0].tolist()]
        if return_subset and subset_size!=0:
            indices = np.arange(len(labels))
            np.random.shuffle(indices)
            subset_indices = indices[:50]
            metadata_labels = [label_mapping[l] for l in labels]
            embeddings = torch.index_select(embeddings, 0, torch.tensor(subset_indices, device=embeddings.device))
            labels = [metadata_labels[i] for i in subset_indices]
            images = torch.index_select(images, 0, torch.tensor(subset_indices, device=images.device))

    backbone.train()
    return embeddings, images, labels



