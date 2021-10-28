import sys
import yaml

import helper

def read_params(file):
    with open(file) as f:
        args = yaml.safe_load(f)

    helper.print_seperate_line()
    print('ARGS ARE: ')
    print(args)
    helper.print_seperate_line()
    return args

def train_process(args, start_training=True):
    print('Start preparing for training ')
    # TODO: Set up data loader with augmentations

    # TODO: Build the student and teacher networks

    # TODO: Set up the training procedure

    return

def main(params_file):
    # Read params and print them
    args = read_params(file=params_file)

    # Set up training
    train_process(args, start_training=True)


if __name__ == '__main__':
    main(params_file='yaml/test_params.yaml')