import os
import sys
import torch
from main import knn_with_features, parse_args
import helper
import warnings

warnings.filterwarnings("ignore")

def main(rank, args):
    feature_folder = '/storage/yue/dino_models/test/'
    save_params = args['save_params']
    train_embeddings = torch.load(os.path.join(feature_folder, "trainembeddings0003.pth"))
    val_embeddings = torch.load(os.path.join(feature_folder, "valembeddings0003.pth"))
    train_labels = torch.load(os.path.join(feature_folder, "trainlabels0003.pth"))
    val_labels = torch.load(os.path.join(feature_folder, "vallabels0003.pth"))
    if rank==0:
        knn_with_features(writer=None, train_embeddings=train_embeddings, train_labels=train_labels,
                          val_embeddings=val_embeddings, val_labels=val_labels, epoch=0, save_params=save_params, if_original=False, if_eval=False)

if __name__ == '__main__':
    args = parse_args(params_path='yaml/test_params.yaml')
    # Launch multi-gpu / distributed training
    helper.launch(main, args)