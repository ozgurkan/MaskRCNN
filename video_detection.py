import os
import sys
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import cv2
import argparse

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "/content/drive/My Drive/logs")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

sys.path.append(os.path.join(ROOT_DIR, "../"))

class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Demo Purpose"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
     

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6
config = CustomConfig()
config.display()

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


# based on https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
# and https://github.com/markjay4k/Mask-RCNN-series/blob/887404d990695a7bf7f180e3ffaee939fbd9a1cf/visualize_cv.py
def display_instances(image, boxes, masks, class_ids, class_names, scores=None):
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
    N = boxes.shape[0]

    colors = colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]

    for i, c in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        
        y1, x1, y2, x2 = boxes[i]
        label = class_names[class_ids[i]]
        score = scores[i] if scores is not None else None
        caption = "{} {:.3f}".format(label, score) if score else label

        # Mask
        mask = masks[:, :, i]
        image = apply_mask(image, mask, c)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), c, 2)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, c, 2)
    return image


class InferenceConfig(CustomConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MaskRCNN Video Object Detection/Instance Segmentation')
    parser.add_argument('-v', '--video_path', type=str, default='', help='Path to video. If None camera will be used')
    parser.add_argument('-sp', '--save_path', type=str, default='', help= 'Path to save the output. If None output won\'t be saved')
    parser.add_argument('-s', '--show', default=True, action="store_false", help='Show output')
    args = parser.parse_args()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "/content/drive/My Drive/logs/sperm_detection20201024T1742/mask_rcnn_sperm_detection_0007.h5")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "/content/drive/My Drive/logs/sperm_detection20201024T1742/mask_rcnn_sperm_detection_0007.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    class_names = ['BG', 'not sperm','sperm']
    
    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=DEFAULT_LOGS_DIR)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    if args.video_path != '':
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)

    if args.save_path:
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
    while cap.isOpened():
        ret, image = cap.read()
        results = model.detect([image], verbose=1)
        r = results[0]
        image = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        if args.show:
            #cv2.imshow('MaskRCNN Object Detection/Instance Segmentation', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if args.save_path:
            out.write(image)
    cap.release()
    if args.save_path:
        out.release()
    #cv2.destroyAllWindows()
