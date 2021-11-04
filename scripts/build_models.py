import torch
from torchvision import models as torchvision_models
import torch.nn as nn

import vit_models as vits
from vit_models import DINOHead
import helper

class DINO(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, rank, model_params):
        super(DINO, self).__init__()

        self.rank = rank
        if model_params['backbone_option'] in vits.__dict__.keys():
            self.student = vits.__dict__[model_params['backbone_option']](
                patch_size=model_params['patch_size'],
                drop_path_rate=model_params['drop_path_rate'],  # stochastic depth
            )
            self.teacher = vits.__dict__[model_params['backbone_option']](patch_size=model_params['patch_size'])
            self.embed_dim = self.student.embed_dim
        elif model_params['backbone_option'] in torchvision_models.__dict__.keys():
            self.student = torchvision_models.__dict__[model_params['backbone_option']]()
            self.teacher = torchvision_models.__dict__[model_params['backbone_option']]()
            self.embed_dim = self.student.fc.weight.shape[1]

        # disable layers dedicated to ImageNet labels classification
        self.student.fc, self.student.head = nn.Identity(), nn.Identity()
        self.teacher.fc, self.teacher.head = nn.Identity(), nn.Identity()

        self.student_head = DINOHead(
            self.embed_dim,
            model_params['out_dim'],
            use_bn=model_params['use_bn_in_head'],
            norm_last_layer=model_params['norm_last_layer'],
        )
        self.teacher_head = DINOHead(
            self.embed_dim,
            model_params['out_dim'],
            use_bn=model_params['use_bn_in_head'])

        # helper.print_layers(self.teacher)

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, student_output, teacher_output = 0, torch.empty(0).to(x[0].device), torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            student_out = self.student(torch.cat(x[start_idx: end_idx]))
            teacher_out = self.teacher(torch.cat(x[start_idx: end_idx]))

            # accumulate outputs
            student_out = torch.cat((student_output, student_out))
            teacher_out = torch.cat((teacher_output, teacher_out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.student_head(student_out), self.teacher_head(teacher_out)



