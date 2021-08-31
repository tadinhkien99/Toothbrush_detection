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
        #mask_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eroded_img = cv2.erode(vismask, kernel = np.ones((9, 9), np.uint8), iterations=3)
        dilated_img = cv2.dilate(eroded_img, kernel = np.ones((7, 7), np.uint8), iterations=3)
        cv2.imshow("Dilated image", dilated_img)
        _, contours, _ = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        center = []
        cnt = []
        angles = []
        sz = []
        for cont in contours:
            if cv2.contourArea(cont) > 500:
                cnt.append(cv2.contourArea(cont))
                rect = cv2.minAreaRect(cont)
                (x, y), (w, h), angle = cv2.minAreaRect(cont)
                center.append(x)
                center.append(y)
                angles.append(angle)
                sz.append(w)
                sz.append(h)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img_cp_1, [box], 0, (0, 0, 255), 2)
        (startY1, startX1, endY1, endX1) = r["rois"][i]

        if cnt[0] > cnt[1]:
            img_cp = cv2.arrowedLine(img_cp, (int(center[0]), int(center[1])), (int(center[2]), int(center[3])),
                                    (0,255,0), 2)
            if center[1] > center [3]:
                if sz[0] < sz[1]:
                    angles[0] = angles[0] - 90
                cv2.putText(image, "Angle ={:.1f} degree".format(angles[0]),
                        (startX1 , startY1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,128), 2)
            elif center[1] < center [3]:
                angle = angle - 180
                if sz[0] < sz[1]:
                    angles[0] = angles[0] - 90
                cv2.putText(image, "Angle ={:.1f} degree".format(angles[0]),
                (startX1, startY1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,128), 2)
        else:
            img_cp = cv2.arrowedLine(img_cp, (int(center[2]), int(center[3])), (int(center[0]), int(center[1])),
                                 (0, 255, 0), 2)
            if center[1] > center [3]:
                angles[1] = angles[1] - 180
                if sz[2] < sz[3]:
                    angles[1] = angles[1] - 90
                cv2.putText(image, "Angle ={:.1f} degree".format(angles[1]),
                    (startX1 , startY1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,128), 2)
            elif center[1] < center [3]:
                if sz[2] < sz[3]:
                    angles[1] = angles[1] - 90
                cv2.putText(image, "Angle ={:.1f} degree".format(angles[1]),
                (startX1, startY1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,128), 2)

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

