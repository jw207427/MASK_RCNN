# load modules
import argparse, os, shutil
import numpy as np
import codecs
import json

import tensorflow as tf
import boto3
    
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# Kangaroo Dataset classes
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset


# model starts here
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    
    # load all the parameters
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--weights', type=str, default=os.environ['SM_CHANNEL_WEIGHTS'])
    parser.add_argument('--hyperparameters', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    weights_file = args.weights
    hyperparameters = json.loads(parser.hyperparameters)
    
    print('gpu count: {}'.format(gpu_count))
    print('model directory: {}'.format(model_dir))
    print('train directory: {}'.format(training_dir))
    print('validation directory: {}'.format(validation_dir))
    print('weights file: {}'.format(weights_file))
    print('hyperparameters: ', hyperparameters)
                