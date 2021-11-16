import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import helper
from helper import accuracy

@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool, model_params):
    linear_classifier.eval()
    metric_logger = helper.MetricLogger(delimiter="  ")
    header = 'Val:'
    for inp, target, _ in metric_logger.log_every(val_loader, 20, header):
        # Move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # Forward
        with torch.no_grad():
            if "vit" in model_params['backbone_option']:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, = accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
    
@torch.no_grad()
def compute_embedding(backbone, dataloader, label_mapping, use_cuda=False, return_tb=False, subset_size=0):
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
        if helper.is_main_process() and embeddings is None:
            embeddings = torch.zeros(len(dataloader.dataset), embs.shape[-1])
            if use_cuda:
                embeddings = embeddings.cuda(non_blocking=True)
            # print(f"Storing features into tensor of shape {embeddings.shape}")
        if return_tb:
            if helper.is_main_process() and images is None:
                images = torch.zeros(len(dataloader.dataset), img.shape[1], img.shape[2], img.shape[3])
                if use_cuda:
                    images = images.cuda(non_blocking=True)
                # print(f"Storing images into tensor of shape {images.shape}")

        if helper.is_main_process() and labels is None:
            labels = torch.zeros(len(dataloader.dataset), lab.shape[-1])
            if use_cuda:
                labels = labels.cuda(non_blocking=True)
            # print(f"Storing labels into tensor of shape {labels.shape}")

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

        if return_tb:
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
        if helper.is_main_process():
            if use_cuda:
                embeddings.index_copy_(0, index_all, torch.cat(embs_l))
                labels.index_copy_(0, index_all, torch.cat(lab_l))
                if return_tb:
                    images.index_copy_(0, index_all, torch.cat(img_l))
            else:
                embeddings.index_copy_(0, index_all.cpu(), torch.cat(embs_l).cpu())
                labels.index_copy_(0, index_all.cpu(), torch.cat(lab_l).cpu())
                if return_tb:
                    images.index_copy_(0, index_all.cpu(), torch.cat(img_l).cpu())


    if helper.is_main_process():
        # if return_tb for tensorboard logging, returned labels contain real cls names such as "golf ball" etc
        # Otherwise, return dataset labels [0, 1, ...]
        if return_tb and subset_size!=0:
            labels = [dataloader.dataset.classes[int(i)] for i in labels.view(1,-1)[0].tolist()]
            indices = np.arange(len(labels))
            np.random.shuffle(indices)
            subset_indices = indices[:subset_size]
            metadata_labels = [label_mapping[l] for l in labels]
            embeddings = torch.index_select(embeddings, 0, torch.tensor(subset_indices, device=embeddings.device))
            labels = [metadata_labels[i] for i in subset_indices]
            images = torch.index_select(images, 0, torch.tensor(subset_indices, device=images.device))
    backbone.train()
    if return_tb:
        return embeddings, images, labels
    else:
        return embeddings, labels

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
                   idx : min((idx + imgs_per_chunk), num_test_images), :
                   ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5




