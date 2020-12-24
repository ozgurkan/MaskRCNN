# import libs
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io
import pylab
import random
from tqdm import tqdm
import time
import math
import re


import xml.etree.ElementTree as ET


# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import model as modellib
from mrcnn import visualize
from mrcnn.model import log
import model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join('./weights', "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "/content/drive/My Drive/logs")   # if you used local system so you need to change this path


############################################################
#  Configurations
############################################################

category = 'sperm_detection'
class_names = ['not sperm','sperm']

class CustomConfig(Config):
    # Give the configuration a recognizable name
    NAME = category
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    PRE_NMS_LIMIT=6000 
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + len(class_names)  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6
config = CustomConfig()
config.display()

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        # Add classes. We have only one class to add.
        for i in range(1,len(class_names)+1):
            self.add_class(category, i, class_names[i-1])

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        

        coco = COCO(os.path.join(dataset_dir, 'annotations.json'))

        category_ids = coco.loadCats(coco.getCatIds())
        cat_ids = [obj['id'] for obj in category_ids]


        for catIndex in tqdm(cat_ids):
            image_ids = coco.getImgIds(catIds=catIndex) 
            for i in range(len(image_ids)):
                polygons = []
                class_ids = []

                tmp = coco.loadImgs([image_ids[i]])
                w,h = tmp[0]['width'],tmp[0]['height']
                a_image_id = image_ids[i]
                img = coco.loadImgs(a_image_id)[0] #here fetching it
                annotation_ids = coco.getAnnIds(imgIds=img['id'])
                annotations = coco.loadAnns(annotation_ids)    
                image_path = os.path.join(dataset_dir,'images/'+img['file_name'])

                #we are now inputting the polygons
                for j in range(len(annotations)):
                    all_points_x = []
                    all_points_y = []

                    for n in range(0,len(annotations[j]['segmentation'][0]),2):
                        all_points_x.append(annotations[j]['segmentation'][0][n])
                        all_points_y.append(annotations[j]['segmentation'][0][n+1])
                    polygons.append({'name':'polygon', 'all_points_x':all_points_x, 'all_points_y':all_points_y})
                    idx = cat_ids.index(annotations[j]['category_id'])
                    class_ids.append(class_names.index(category_ids[idx]['name'])+1)

                self.add_image(
                    category,  ## for a single class just add the name here
                    image_id=img['file_name'],  # use file name as a unique image id
                    path=image_path,
                    width=w, height=h,
                    polygons=polygons, 
                    class_ids=class_ids)

    #TOMODIFY FROM HERE
    def load_mask(self, image_id):
        # If not a bottle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != category:
            return super(self.__class__, self).load_mask(image_id)
        
        
        class_ids = image_info['class_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. 
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == category:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
      image = dataset_train.load_image(image_id)
      mask, class_ids = dataset_train.load_mask(image_id)
      visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

      # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last
    if init_with == "imagenet":
       model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
      # Load weights trained on MS COCO, but skip layers that
      # are different due to the different number of classes
      # See README for instructions to download the COCO weights
      model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
      # Load the last model you trained and continue training
      model.load_weights(model.find_last(), by_name=True)

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    
    #mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model,model_inference, dataset_val, calculate_map_at_every_X_epoch=5, verbose=1)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,                
                epochs=40,
                layers='heads',
                custom_callbacks=None)
                #custom_callbacks=[mean_average_precision_callback])


    class InferenceConfig(CustomConfig):
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()                  

      # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=DEFAULT_LOGS_DIR)
    MaxMap=0.0
    for i in range(1,41):           
        # Get path to saved weights
        # Either set a specific path or find last trained weights
        if i<10:
          model_path = os.path.join(ROOT_DIR, "/content/drive/My Drive/logs/sperm_detection20201024T1742/mask_rcnn_sperm_detection_000{}.h5".format(i))
          #model_path = model.find_last()
        else:
          model_path = os.path.join(ROOT_DIR, "/content/drive/My Drive/logs/sperm_detection20201024T1742/mask_rcnn_sperm_detection_00{}.h5".format(i))

        # Load trained weights
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)
        # Compute VOC-Style mAP @ IoU=0.5
        # Running on 10 images. Increase for better accuracy.
        image_ids = dataset_val.image_ids
        APs = []
        ARs = [] 
        F1_scores = []
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_val, inference_config,
                                      image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            AP, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                r["rois"], r["class_ids"], r["scores"], r['masks'])

            APs.append(AP)

            AR, positive_ids = utils.compute_recall(r["rois"], gt_bbox, iou=0.2)
            ARs.append(AR)
            F1_scores.append((2* (np.mean(precisions) * np.mean(recalls)))/(np.mean(precisions) + np.mean(recalls)))
        print("Precision: ", np.mean(precisions))
        print("Recall: ",np.mean(recalls))
        print("Overlap: ",np.mean(overlaps))

        print("mAP: ", np.mean(APs))
        print("mAR: ", np.mean(ARs))
        print("F1_scores: ", np.mean(F1_scores))           
        if MaxMap<np.mean(APs):
          MaxMap=np.mean(APs)
    
        mAP, mAR, F1_score = evaluate_model(dataset_val, model, inference_config)
        print("mAP: %.3f" % mAP)
        print("mAR: %.3f" % mAR)
        print("first way calculate f1-score: ", F1_score)

        F1_score_2 = (2 * mAP * mAR)/(mAP + mAR)
        print('second way calculate f1-score_2: ', F1_score_2) 
    print("MAX MAP:%f",MaxMap)   


import numpy as np
def evaluate_model(dataset, model, cfg):
        APs = list();
        ARs = list(); 
        F1_scores = list(); 
        for image_id in dataset.image_ids:
            image, image_meta, gt_class_id, gt_bbox, gt_mask =modellib.load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
            scaled_image =modellib.mold_image(image, cfg)
            sample = np.expand_dims(scaled_image, 0)          
            yhat = model.detect(sample, verbose=0)
            r = yhat[0]
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
            AR, positive_ids = utils.compute_recall(r["rois"], gt_bbox, iou=0.2)
            ARs.append(AR)
            F1_scores.append((2* (np.mean(precisions) * np.mean(recalls)))/(np.mean(precisions) + np.mean(recalls)))
            APs.append(AP)
        
        mAP = np.mean(APs)
        mAR = np.mean(ARs)
        return mAP, mAR, F1_scores
#######################################################################################################################



def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
