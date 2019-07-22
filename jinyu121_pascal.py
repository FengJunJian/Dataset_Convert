"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained weights
    python3 jinyu121_pascal.py train --dataset=/path/to/coco/ --model=voc

    # Train a new model starting from ImageNet weights
    python3 jinyu121_pascal.py train --dataset=/path/to/voc/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 jinyu121_pascal.py train --dataset=/path/to/voc/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 jinyu121_pascal.py train --dataset=/path/to/voc/ --model=last

    # Run evaluatoin on the last model you trained
    python3 jinyu121_pascal.py evaluate --dataset=/path/to/voc/ --model=last
"""

import os
import time
import xmltodict
import numpy as np
import skimage
import json
from tqdm import tqdm

from config import Config
import utils
import model as modellib

from voc_eval import voc_ap

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
VOC_MODEL_PATH = os.path.join(ROOT_DIR, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class VOCConfig(Config):
    """Configuration for training on PASCAL VOC.
    Derives from the base Config class and overrides values specific
    to the PASCAL VOC dataset.
    """
    # Give the configuration a recognizable name
    NAME = "voc"

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 2

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 150

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # VOC has 20 classes


############################################################
#  Dataset
############################################################

class VOCDataset(utils.Dataset):
    def load_voc(self, dataset_dir, mode, class_ids=None, class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, val35k)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """
        self.stop_color = [np.array([0, 0, 0]), np.array([224, 224, 192])]
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow",
                        "diningtable", "dog", "horse", "motorbike", "person",
                        "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.items_count = {c: 0 for c in self.classes}
        for ith, cls in enumerate(self.classes):
            self.add_class("voc", ith + 1, cls)

        # Path
        split_dir = os.path.join(dataset_dir, 'ImageSets', 'Segmentation')
        annotation_dir = os.path.join(dataset_dir, 'Annotations')
        image_dir = os.path.join(dataset_dir, "JPEGImages")
        mask_dir = os.path.join(dataset_dir, "SegmentationObject")

        # Load split
        dataset_txt = os.path.join(split_dir, mode + ".txt")

        for ith, line in enumerate(open(dataset_txt)):
            line = line.strip()
            with open(os.path.join(annotation_dir, line + ".xml")) as f:
                anno = xmltodict.parse(f.read())
                objs = anno['annotation']['object']
                if not isinstance(objs, list):
                    objs = [objs]
                for o in objs:
                    self.items_count[o['name']] += 1
            self.add_image(
                source='voc',
                image_id=ith,
                path=os.path.join(image_dir, line + '.jpg'),
                mask_path=os.path.join(mask_dir, line + '.png'),
                objs=objs,
                width=anno['annotation']["size"]['width'],
                height=anno['annotation']["size"]['height'],
            )

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        center_points = []
        for ith in image_info['objs']:
            x = (int(ith['bndbox']['xmin']) + int(ith['bndbox']['xmax'])) / 2
            y = (int(ith['bndbox']['ymin']) + int(ith['bndbox']['ymax'])) / 2
            center_points.append([x, y])
        n_objs = len(center_points)
        used_flag = [False] * n_objs
        center_points = np.array(center_points)
        mask_image = skimage.io.imread(image_info['mask_path'])
        colors = np.unique(mask_image.reshape([-1, 3]), axis=0)
        for ith in range(colors.shape[0]):
            color = colors[ith]
            if np.all(color == self.stop_color[0]) or np.all(color == self.stop_color[1]):
                pass
            else:
                # Make mask
                tmp_mask = mask_image == color
                # Merge 3 channels mask to one channel
                mask = np.ones(tmp_mask.shape[:2]).astype(np.bool)
                for l in range(3):
                    mask = np.logical_and(mask, tmp_mask[:, :, l])

                instance_masks.append(mask)

                # Find center of object
                col = np.where(mask.sum(axis=0) > 0)[0]
                row = np.where(mask.sum(axis=1) > 0)[0]

                center_x = (col[0] + col[-1]) / 2
                center_y = (row[0] + row[-1]) / 2

                # Assign class label to object
                near = np.argsort(
                    np.sum(
                        np.power(center_points - np.array([center_x, center_y]), 2),
                        axis=1
                    )
                )
                for sel in near:
                    if not used_flag[sel]:
                        used_flag[sel] = True
                        class_id = self.classes.index(image_info['objs'][sel]['name']) + 1
                        class_ids.append(class_id)
                        break
                else:
                    raise AssertionError("Segmentation part more than location objects")
        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Pascal VOC.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on VOC")
    parser.add_argument('--dataset',
                        metavar="/path/to/coco/",
                        default="/home/Datasets/PASCALVOC/VOCdevkit/VOC2012",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model',
                        metavar="/path/to/weights.h5",
                        default="init_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                        help="Path to weights .h5 file")
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)

    # Configurations
    if args.command == "train":
        config = VOCConfig()
    else:
        class InferenceConfig(VOCConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=MODEL_DIR)

    # Select weights file to load
    if args.model.lower() == "voc":
        model_path = VOC_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = VOCDataset()
        dataset_train.load_voc(args.dataset, "train")
        dataset_train.prepare()
        print("Items count in training set")
        print(json.dumps(dataset_train.items_count, indent=4))

        dataset_val = VOCDataset()
        dataset_val.load_voc(args.dataset, "val")
        dataset_val.prepare()
        print("Items count in evaluate set")
        print(json.dumps(dataset_val.items_count, indent=4))

        # This training schedule is an example. Update to fit your needs.

        # Training - Stage 1
        # Adjust epochs and layers as needed
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Training Resnet layer 4+")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=100,
                    layers='4+')

        # Training - Stage 3
        # Finetune layers from ResNet stage 3 and up
        print("Training Resnet layer 3+")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 100,
                    epochs=200,
                    layers='all')

    elif args.command == "evaluate":
        dataset_val = VOCDataset()
        dataset_val.load_voc(args.dataset, "val")
        dataset_val.prepare()
        print("Items count in evaluate set")
        print(json.dumps(dataset_val.items_count, indent=4))

        correct_list = []
        score_list = []
        overlap_list = []

        for image_id in tqdm(dataset_val.image_ids):
            # Load image and ground truth data
            image, image_meta, class_ids, gt_bbox, gt_mask = modellib.load_image_gt(
                dataset_val, config, image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, config), 0)

            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]

            # Compute right or wrong
            co, sc, ov = utils.compute_mask_right_wrong(
                gt_bbox[:, :4], gt_mask, class_ids,
                r['rois'], r["masks"], r['class_ids'],
                r["scores"], overlap_thresh=0.5)

            # Merge them together
            correct_list.append(co)
            score_list.append(sc)
            overlap_list.append(ov)

        right = np.concatenate(correct_list).ravel()
        wrong = np.logical_not(right)
        score_list = np.concatenate(score_list).ravel()
        overlap_list = np.concatenate(overlap_list).ravel()

        order = (-score_list).argsort()

        tp = np.cumsum(right[order].astype(int))
        fp = np.cumsum(wrong[order].astype(int))
        rec = tp / float(len(order))
        prec = tp / (tp + fp)

        ap = voc_ap(rec, prec, False)
        mean_iou = overlap_list.sum() / float(tp[-1])

        print('Accuracy:\t{}'.format(tp[-1] / len(tp)))
        print('AP:\t{}'.format(ap))
        print('Mean IoU:\t{}'.format(mean_iou))

    else:
        print("'{}' is not recognized. Use 'train' or 'evaluate'".format(args.command))
