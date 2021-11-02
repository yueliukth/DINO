import vit_models as vits
from torchvision import models as torchvision_models

def get_dino(model_params):
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


    return student, teacher