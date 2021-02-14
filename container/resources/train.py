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

# class that defines and loads the kangaroo dataset
class KangarooDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir):
        # define one class
        self.add_class("dataset", 1, "kangaroo")
        # define data locations
        images_dir = dataset_dir + '/image/'
        annotations_dir = dataset_dir + '/annots/'
        # find all images
        for filename in os.listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('kangaroo'))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
    
# define a configuration for the model
class KangarooConfig(Config):
    # define the name of the configuration
    NAME = "kangaroo_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = 131

# model starts here
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    
    # load all the parameters
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--weights', type=str, default=os.environ['SM_CHANNEL_WEIGHTS'])
    
    # hyperparameters
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128) 
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--weights-file', type=str, default='mask_rcnn_coco.h5')
    
    args, _ = parser.parse_known_args()
    
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    weights_dir = args.weights
    
    # hyperparameters
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    num_classes = args.num_classes
    weights_file = args.weights_file
    
    print('gpu count: {}'.format(gpu_count))
    print('model directory: {}'.format(model_dir))
    print('train directory: {}'.format(training_dir))
    print('validation directory: {}'.format(validation_dir))
    print('weights file: {}'.format(weights_file))
    
    # prepare train set
    train_set = KangarooDataset()
    train_set.load_dataset(training_dir)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    # prepare test/val set
    test_set = KangarooDataset()
    test_set.load_dataset(validation_dir)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
    
        # prepare config
    config = KangarooConfig()
    config.NAME = "kangaroo_training"
    config.NUM_CLASSES = num_classes
    config.GPU_COUNT = gpu_count
    config.STEPS_PER_EPOCH = batch_size
    config.LEARNING_RATE = lr
    
    config.display()
    
    # define the model
    model = MaskRCNN(mode='training', model_dir=model_dir, config=config)
    # load weights (mscoco) and exclude the output layers
    model.load_weights(os.path.join(weights_dir, weights_file), 
                       by_name=True, 
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    # train weights (output layers or 'heads')
    model.train(train_set, 
                test_set, 
                learning_rate=config.LEARNING_RATE, 
                epochs=epochs, 
                layers='heads')
    
    output_path = model.find_last()
    
    try:
        output = shutil.copy(output_path, model_dir)
        print(f'model artifacts successfully saved in {output}')
    except Exception as e:
        print( "Error: %s" % str(e))