import torch
from torchvision import models as torchvision_models
import torch.nn as nn

import vit_models as vits
from vit_models import trunc_normal_
import helper

def build_dino(model_params):
    if model_params['backbone_option'] in vits.__dict__.keys():
        student = vits.__dict__[model_params['backbone_option']](
            patch_size=model_params['patch_size'],
            drop_path_rate=model_params['drop_path_rate'],  # stochastic depth
        )
        teacher = vits.__dict__[model_params['backbone_option']](patch_size=model_params['patch_size'])
        embed_dim = student.embed_dim
    elif model_params['backbone_option'] in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[model_params['backbone_option']]()
        teacher = torchvision_models.__dict__[model_params['backbone_option']]()
        embed_dim = student.fc.weight.shape[1]

    # Disable layers dedicated to ImageNet labels classification
    student.fc, student.head = nn.Identity(), nn.Identity()
    teacher.fc, teacher.head = nn.Identity(), nn.Identity()

    student_head = Head(
        embed_dim,
        model_params['out_dim'],
        use_bn=model_params['use_bn_in_head'],
        norm_last_layer=model_params['norm_last_layer'],
    )
    teacher_head = Head(
        embed_dim,
        model_params['out_dim'],
        use_bn=model_params['use_bn_in_head'])
    return student, student_head, teacher, teacher_head

class Head(nn.Module):
    """Network hooked up to the CLS token embedding.
    Just a MLP with the last layer being normalized in a particular way.
    Parameters
    ----------
    in_dim : int
        The dimensionality of the token embedding.
    out_dim : int
        The dimensionality of the final layer (we compute the softmax over).
    use_bn: bool
        If True, then we use batch norm.
    norm_last_layer : bool
        If True, then we freeze the norm of the weight of the last linear layer
        to 1.
    nlayers : int
        The number of layers.
    hidden_dim : int
        Dimensionality of the hidden layers.
    bottleneck_dim : int
        Dimensionality of the second last layer.

    Attributes
    ----------
    mlp : nn.Sequential
        Vanilla multi-layer perceptron.
    last_layer : nn.Linear
        Reparametrized linear layer with weight normalization. That means
        that that it will have `weight_g` and `weight_v` as learnable
        parameters instead of a single `weight`.
    """
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # PyTorch weight norm is done with nn.utils.weight_norm that returns weight_g and weight_v
        # Original weights can be recovered by
        # weight_g * (weight_v / weight_v.norm(dim=-1, keepdim=True))
        # Weight normalised module has _forward_pre_hooks while original module does not have
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # Freeze the magnitude
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        """Initialize learnable parameters."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Of shape `(n_samples, in_dim)`.
        Returns
        -------
        torch.Tensor
            Of shape `(n_samples, out_dim)`.
        """
        x = self.mlp(x) # (n_samples, bottleneck_dim)
        x = nn.functional.normalize(x, dim=-1, p=2) # (n_samples, bottleneck_dim)
        x = self.last_layer(x) # (n_samples, out_dim)
        return x

class MultiCropWrapper(nn.Module):
    """Convenience class for forward pass of multiple crops.
    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Instantiated Vision Transformer. Note that we will take the `head`
        attribute and replace it with `nn.Identity`.
    new_head : Head
        New head that is going to be put on top of the `backbone`.
    """
    def __init__(self, backbone, head):
        super().__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity() # disable original head
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        """Run the forward pass.
        The different crops are concatenated along the batch dimension
        and then a single forward pass is fun. The resulting tensor
        is then chunked back to per crop tensors.
        Parameters
        ----------
        x :
            Input and if not list, converted to list of `torch.Tensor` each of shape `(n_samples, nc, size, size)`.
        Returns
        -------
        tuple
            Tuple of `torch.Tensor` each of shape `(n_samples, out_dim)` where
            `output_dim` is determined by `Head`.
        """
        # Convert to list
        if not isinstance(x, list):
            x = [x]

        _, counts = torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]), return_counts=True,)
        idx_crops = torch.cumsum(counts, 0)

        start_idx, output = 0, torch.empty(0).to(x[0].device)

        for end_idx in idx_crops:
            out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # accumulate outputs
            output = torch.cat((output, out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        head_output = self.head(output)
        return head_output.chunk(len(x))








