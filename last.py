# USAGE
# python maskrcnn_predict.py --weights mask_rcnn_coco.h5 --labels coco_labels.txt --image train/30th_birthday.jpg

# import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
    help="path to Mask R-CNN model weights pre-trained on COCO")
ap.add_argument("-l", "--labels", required=True,
    help="path to class labels file")
ap.add_argument("-i", "--image", required=True,
    help="path to input image to apply Mask R-CNN to")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())

# load the class label names from disk, one label per line
CLASS_NAMES = open(args["labels"]).read().strip().split("\n")

# generate random (but visually distinct) colors for each class label
# (thanks to Matterport Mask R-CNN for the method!)
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)


image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = imutils.resize(image, width=512)
image = imutils.resize(image, width=560)
img_cp = image.copy()
img_cp_1 = image.copy()
img_cp_2 = image.copy()
length_arrow = 50
split = 11
def detection(image, img_cp):
    # perform a forward pass of the network to obtain the results
    print("Making predictions with Mask R-CNN...")
    r = model.detect([image], verbose=1)[0]
    max = r["masks"][:, :, 0]
    # loop over of the detected object's bounding boxes and masks
    for i in range(0, r["rois"].shape[0]):
        # extract the class ID and mask for the current detection, then
        # grab the color to visualize the mask (in BGR format)
        classID = r["class_ids"][i]
        mask = r["masks"][:, :, i]
        color = COLORS[classID][::-1]
        # visualize the pixel-wise mask of the object
        img_cp = visualize.apply_mask(img_cp, mask, color, alpha=0.9)
        if mask.any() > max.all():
            max = mask
            vismax = (max * 255).astype("uint8")
            a=i
        cv2.imshow("max_mask", vismax)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_cp = cv2.cvtColor(img_cp, cv2.COLOR_RGB2BGR)
    # loop over the predicted scores and class labels
    for i in range(0, len(r["scores"])):
        # extract the bounding box information, class ID, label, predicted
        # probability, and visualization color
        (startY, startX, endY, endX) = r["rois"][i]
        classID = r["class_ids"][i]
        label = CLASS_NAMES[classID]
        score = r["scores"][i]
        print("Accuracy = ({}: {})\n".format(i, score))
        color = [int(c) for c in np.array(COLORS[classID]) * 255]

        # draw the bounding box, class label, and score of the object
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        if i ==a:
            text = "{}: {:.3f}--Top".format(label, score)
        else:
            text = "{}: {:.3f}".format(label, score)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,0.6, color, 2)
    vismask =[]

    for i in range(0, r["rois"].shape[0]):
        mask = r["masks"][:,:,i]
        vismask = (mask * 255).astype("uint8")
        _, contours, _ = cv2.findContours(vismask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        length = []
        for cont in contours:
            if cv2.contourArea(cont) > 1000:
                rect = cv2.minAreaRect(cont)
                (x, y), (w, h), angle = cv2.minAreaRect(cont)
                x = int(x)
                y = int(y)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                a1 = math.sqrt(pow(box[0][0] - box[1][0], 2) + pow(box[0][1] - box[1][1], 2))
                a2 = math.sqrt(pow(box[0][0] - box[3][0], 2) + pow(box[0][1] - box[3][1], 2))
                if a1 > a2:
                    angle = math.atan2(box[0][1] - box[1][1], box[0][0] - box[1][0]) * 180 / math.pi
                    angle = -angle
                    for i in range(1, split):
                        masks = np.zeros((img_cp_2.shape[0], img_cp_2.shape[1]), np.uint8)
                        xa = int(box[0][0] - a1 * i * math.cos(math.radians(abs(angle))) / split)
                        ya = int(box[0][1] - a1 * i * math.sin(math.radians(abs(angle))) / split)
                        xb = int(box[3][0] - a1 * i * math.cos(math.radians(abs(angle))) / split)
                        yb = int(box[3][1] - a1 * i * math.sin(math.radians(abs(angle))) / split)
                        cv2.line(masks, (xa, ya), (xb, yb), (255, 255, 255), 2)
                        matching = cv2.bitwise_and(masks, vismask)

                        # find white pixel
                        idx = np.where(matching == [255])
                        length.append(len(idx[0]))
                    top_toothbrush = np.argmin(length)
                    n = top_toothbrush
                    #masks = np.zeros((img_cp_2.shape[0], img_cp_2.shape[1]), np.uint8)
                    xa = int(box[0][0] - a1 * n * math.cos(math.radians(abs(angle))) / split)
                    ya = int(box[0][1] - a1 * n * math.sin(math.radians(abs(angle))) / split)
                    xb = int(box[3][0] - a1 * n * math.cos(math.radians(abs(angle))) / split)
                    yb = int(box[3][1] - a1 * n * math.sin(math.radians(abs(angle))) / split)
                    cv2.line(vismask, (xa, ya), (xb, yb), (0, 0, 0), 2)
                    #matching = cv2.bitwise_or(masks, vismask)
                    cv2.imshow('ttt',vismask)
                else:
                    angle = math.atan2(box[0][1] - box[3][1], box[3][0] - box[0][0]) * 180 / math.pi
                    for i in range(1, split):
                        masks = np.zeros((img_cp_2.shape[0], img_cp_2.shape[1]), np.uint8)
                        xa = int(box[0][0] + a2 * i * math.cos(math.radians(abs(angle))) / split)
                        ya = int(box[0][1] - a2 * i * math.sin(math.radians(abs(angle))) / split)
                        xb = int(box[1][0] + a2 * i * math.cos(math.radians(abs(angle))) / split)
                        yb = int(box[1][1] - a2 * i * math.sin(math.radians(abs(angle))) / split)
                        cv2.line(masks, (xa, ya), (xb, yb), (255, 255, 255), 2)
                        matching = cv2.bitwise_and(masks, vismask)

                        # find white pixel
                        idx = np.where(matching == [255])
                        length.append(len(idx[1]))
                    top_toothbrush = np.argmin(length)
                    n = top_toothbrush
                    #masks = np.zeros((img_cp_2.shape[0], img_cp_2.shape[1]), np.uint8)
                    xa = int(box[0][0] + a2 * n * math.cos(math.radians(abs(angle))) / split)
                    ya = int(box[0][1] - a2 * n * math.sin(math.radians(abs(angle))) / split)
                    xb = int(box[1][0] + a2 * n * math.cos(math.radians(abs(angle))) / split)
                    yb = int(box[1][1] - a2 * n * math.sin(math.radians(abs(angle))) / split)
                    cv2.line(vismask, (xa, ya), (xb, yb), (0, 0, 0), 2)
                    #matching = cv2.bitwise_and(masks, vismask)
                    cv2.imshow('ttt',vismask)
                print(top_toothbrush, ',', length)
                # if cy1 < cy:
                if top_toothbrush == 5 or top_toothbrush == 6 or top_toothbrush == 7 or top_toothbrush == 8 or top_toothbrush == 9:

                    if angle > 0:
                        real_angle = angle
                        P2x = int(x + length_arrow * math.cos(math.radians(abs(angle))))
                        P2y = int(y - length_arrow * math.sin(math.radians(abs(angle))))
                        cv2.arrowedLine(image, (x, y), (P2x, P2y), (0, 255, 0), 3, tipLength=0.15)
                    else:
                        real_angle = angle + 180
                        P2x = int(x - length_arrow * math.cos(math.radians(abs(angle))))
                        P2y = int(y - length_arrow * math.sin(math.radians(abs(angle))))
                        cv2.arrowedLine(image, (x, y), (P2x, P2y), (0, 0, 255), 3, tipLength=0.15)
                else:
                    if angle > 0:
                        real_angle = angle + 180
                        P2x = int(x - length_arrow * math.cos(math.radians(abs(angle))))
                        P2y = int(y + length_arrow * math.sin(math.radians(abs(angle))))
                        cv2.arrowedLine(image, (x, y), (P2x, P2y), (255, 0, 0), 3, tipLength=0.15)
                    else:
                        real_angle = angle + 360
                        P2x = int(x + length_arrow * math.cos(math.radians(abs(angle))))
                        P2y = int(y + length_arrow * math.sin(math.radians(abs(angle))))
                        cv2.arrowedLine(image, (x, y), (P2x, P2y), (25, 25, 25), 3, tipLength=0.15)
                print("X : {}  Y : {} angle: {}".format(x, y, int(real_angle)))

    cv2.imshow("test", img_cp_1)
    cv2.imshow("arrow", img_cp)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey()

if __name__ == '__main__':
    class SimpleConfig(Config):
        # give the configuration a recognizable name
        NAME = "coco_inference"

        # set the number of GPUs to use along with the number of images
        # per GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # number of classes (we would normally add +1 for the background
        # but the background class is *already* included in the class
        # names)
        NUM_CLASSES = len(CLASS_NAMES)

    # initialize the inference configuration
    config = SimpleConfig()
    # initialize the Mask R-CNN model for inference and then load the
    # weights
    print("Loading Mask R-CNN model...")
    #model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='C:/Users/DELL/downloads/keras-mask-rcnn/logs')
    model.load_weights(args["weights"], by_name=True)

    detection(image, img_cp)
    mask_detect()

