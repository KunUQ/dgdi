#Authors: Kun Zhao, Postdoctoral Research Fellow,The University of Queensland, kun.zhao1@uq.edu.au; Danny Smith, Research Manager, The University of Queensland, danny.smith@uq.edu.au.
import os
import sys
import time
import numpy as np
import pyvips
from PIL import Image
import cv2
import tensorflow as tf
#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util

GLOMS_PER_ROW = 4
NUM_CLASSES = 1
OBJECTIVE = "40X"
SCALE = 12
TILE_SIZE_X = 1024
TILE_SIZE_Y = 600
#Define the path for the big_tiff image, trained model and result csv file name
PATH_TO_BIGTIFF = r" "
PATH_TO_CKPT = r" "
CSV_FILE = "gloms.csv"
#PATH_TO_CKPT = r"Z:\Kun\models\rfcn\frozen_inference_graph.pb"
#PATH_TO_BIGTIFF = r"Z:\Kun\sample_data\bigtiff"

def now():
    return time.strftime("%H:%M:%S", time.localtime())

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

#detection function for bigtiff images
def glom_detect(image_np, detection_graph):
    rows = image_np.shape[0]
    cols = image_np.shape[1]
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
# Visualization of the results of a detection.
#    suppressed_boxes = tf.image.non_max_suppression(boxes[0], scores[0], max_output_size=1, iou_threshold=0.5, name=None)
#    vis_util.visualize_boxes_and_labels_on_image_array(
#        image_np,
#        np.squeeze(new_boxes),
#        np.squeeze(classes).astype(np.int32),
#        np.squeeze(new_scores),
#        category_index,
#        use_normalized_coordinates=True,
#        line_thickness=8)
#    return boxes, scores, classes, num_detections
            glom_boxes = []
            for j in range(int(num_detections[0])):
                bbox = [float(v) for v in boxes[0][j]]
                if scores[0][j] > 0.5:
                    left = int(bbox[1] * cols)
                    top = int(bbox[0] * rows)
                    right = int(bbox[3] * cols)
                    bottom = int(bbox[2] * rows)
                    glom_boxes.append([left, top, right, bottom])
                    if any(bbox[i] > 1.0 for i in range(len(bbox))):
                        print ("bbox is too large:", bbox)
            return glom_boxes


if not os.path.isdir(PATH_TO_BIGTIFF):
    print ("Unable to find", PATH_TO_BIGTIFF)
    exit(1)

for in_dir in os.listdir(PATH_TO_BIGTIFF):
#    if in_dir[6:7] not in ["4"]:         #############
#        continue                            #############
    in_file = os.path.join(PATH_TO_BIGTIFF, in_dir, OBJECTIVE, in_dir + "-bigtiff.tif")
    print ("**** Processing", in_file, "at", now())
    if not os.path.isfile(in_file):
        print ("      Unable to find", in_file)
        continue
    if os.path.isfile(os.path.join(PATH_TO_BIGTIFF, in_dir, OBJECTIVE, CSV_FILE)):
        print ("      Already processed - skipping")
        continue

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    bigtiff = pyvips.Image.new_from_file(in_file)
    smalltiff = bigtiff.shrink(SCALE, SCALE)

#    print (bigtiff.width, bigtiff.height, smalltiff.width, smalltiff.height)
    x_ratio = float(bigtiff.width) / float(smalltiff.width)
    y_ratio = float(bigtiff.height) / float(smalltiff.height)
#    print ("Ratios are", round(x_ratio, 2), round(y_ratio, 2))

    gloms = []
    for tile_y in range(0, smalltiff.height, TILE_SIZE_Y // 2):
        for tile_x in range(0, smalltiff.width, TILE_SIZE_X // 2):
            tile_w = (TILE_SIZE_X if (smalltiff.width - tile_x) > TILE_SIZE_X else (smalltiff.width - tile_x))
            tile_h = (TILE_SIZE_Y if (smalltiff.height - tile_y) > TILE_SIZE_Y else (smalltiff.height - tile_y))
            in_patch = smalltiff.crop(tile_x, tile_y, tile_w, tile_h)
            full_patch = pyvips.Image.black(TILE_SIZE_X, TILE_SIZE_Y, bands=3)
            in_patch = full_patch.insert(in_patch, 0, 0, expand=True)
            image_data = np.array(in_patch.write_to_memory()).reshape((in_patch.height, in_patch.width, 3)).astype(np.uint8)
#            print (type(image_data), image_data.shape, image_data.size)
            glom_boxes = []
            if np.count_nonzero(image_data) > 500:
#                image_process, boxes, scores, classes, num_detections = detect_objects(image_np, sess, detection_graph)
                glom_boxes = glom_detect(image_data, detection_graph)
            for glom_box in glom_boxes:
                if glom_box[0] > tile_w or glom_box[1] > tile_h or glom_box[2] > tile_w or glom_box[3] > tile_h:
                    print ("Skipping out of bounds detection", glom_box, tile_x, tile_y, tile_w, tile_h)
                    continue
                glom_box[0] += tile_x
                glom_box[1] += tile_y
                glom_box[2] += tile_x
                glom_box[3] += tile_y
                gloms.append(tuple(glom_box))
#                print (tile_x, tile_y, glom_box)
    print ("      Finished detections at", now())

    big_gloms = non_max_suppression_fast(np.array(gloms), 0.3)

    glom_patches = []
    try:
        csv_file = open(os.path.join(PATH_TO_BIGTIFF, in_dir, OBJECTIVE, CSV_FILE), "w")
    except IOError:
        print ("      Unable to open ", os.path.join(PATH_TO_BIGTIFF, in_dir, OBJECTIVE, CSV_FILE), "- continuing")
        continue
    for glom in big_gloms:
#        print ("Initial glom is", glom)
        glom_l = int(round(glom[0] * x_ratio, 0))
        glom_t = int(round(glom[1] * y_ratio, 0))
        glom_w = int(round(glom[2] * x_ratio, 0)) - glom_l
        if glom_w + glom_l > bigtiff.width:
            glom_w = bigtiff.width - glom_l - 1
        glom_h = int(round(glom[3] * y_ratio, 0)) - glom_t
        if glom_h + glom_t > bigtiff.height:
            glom_h = bigtiff.height - glom_t - 1
#        print ("image patch is", glom_l, glom_t, glom_w, glom_h)
        glom_patches.append(bigtiff.crop(glom_l, glom_t, glom_w, glom_h))
        csv_file.write(str(glom_l) + ", " + str(glom_t) + ", " + str(glom_w) + ", " + str(glom_h) + "\n")
    csv_file.close()

    max_w = max_h = 0
    for glom_patch in glom_patches:
        if glom_patch.width > max_w:
            max_w = glom_patch.width
        if glom_patch.height > max_h:
            max_h = glom_patch.height
    print ("      Finished cropping at", now())

    glom_image = pyvips.Image.black(150, 150, bands=3)
    for i, glom_patch in enumerate(glom_patches):
        glom_image_x = i % GLOMS_PER_ROW
        glom_image_y = i // GLOMS_PER_ROW
        temp_image = glom_image.insert(glom_patch, glom_image_x * max_w, glom_image_y * max_h, expand=True)
        glom_image = temp_image
#        glom_patch.write_to_file(os.path.join(PATH_TO_BIGTIFF, in_file + "_glom" + str(i).zfill(2) + ".tif"))
    glom_image.write_to_file(in_file.replace(".tif", "_gloms.tif"))
    print ("      Finished stitching at", now())

