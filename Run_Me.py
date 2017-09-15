__author__ = 'Vijayabhaskar J'
# Inspired from Sentdex's work on "Python plays GTA V"
# Thanks to him and Daniel Kukiela (https://twitter.com/daniel_kukiela) for the keys.py file
# And also Special thanks to Tensorflow Object Detection API

import keys as k
import time
import cv2
import numpy as np
import os
import tensorflow as tf
from grabscreen import grab_screen
import six.moves.urllib as urllib
import tarfile

keys = k.Keys({})

# ## Object detection imports
# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_OD_DIR=os.path.join(os.getcwd(),"object_detection")
PATH_TO_MODEL=os.path.join(PATH_TO_OD_DIR,MODEL_NAME)
PATH_TO_CKPT = os.path.join(PATH_TO_MODEL,'frozen_inference_graph.pb')
PATH_TO_LABELS=os.path.join(PATH_TO_OD_DIR,"data",'mscoco_label_map.pbtxt')

if os.path.isdir(PATH_TO_OD_DIR):
    if not os.path.isdir(PATH_TO_MODEL):
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
else:
    print("object_detection directory not found")
    raise FileNotFoundError
    sys.exit("object_detection not found")
detection_graph = tf.Graph()

#****************** Parameters to choose ******************

#Resolution of the game(Make sure the game runs in windowed mode in the top left corner of your screen
WIDTH=800
HEIGHT=600

#Fire using Keyboard or Mouse?
#HIGHLY RECOMMENDED TO USE A KEYBOARD KEY TO SHOOT
INPUT="keyboard" #if you want set it to "mouse"

#WHICH KEY SHOULD BE PRESSED INORDER TO FIRE?
#HIGHLY RECOMMENDED TO USE A KEYBOARD KEY TO SHOOT
if INPUT=="keyboard":
    FIRE_KEY="RETURN" #REFER The keys.py file for list of all keys
elif INPUT=="mouse":
    FIRE_KEY=keys.mouse_lb_press #If you want to change look keys.py file
    RELEASE_KEY=keys.mouse_lb_release
#Where to Shoot?
SHOOT="CENTER" #You can use "HEAD" or "NECK" or "CENTER"
# CENTER - Target at 50% of the height of the detected person
# NECK - Target at 25% of the height of the detected person
# HEAD - Target at 12.5% of the height of the detected person

#How long to press the fire button when a person is detected?
HL_TO_FIRE=0.4 #HOLDS THE FIRE BUTTON FOR 0.4 SECONDS

#***********************************************************


def determine_movement(mid_x, mid_y,width=800, height=600):
    x_move = 0.5-mid_x
    y_move = 0.5-mid_y
    keys.keys_worker.SendInput(keys.keys_worker.Mouse(0x0001, -1*int(x_move*width), -1*int(y_move*height)))

with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    print("Running with Parameters:\nDevice: {},WIDTH:{} HEIGHT:{} SHOOT:{} HL_TO_FIRE:{}".format(INPUT,WIDTH,HEIGHT,SHOOT,HL_TO_FIRE))
    while True:
        screen = cv2.resize(grab_screen(region=(0,0,WIDTH,HEIGHT)), (int(WIDTH/4),int(HEIGHT/4)))
        image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=3)
        cv2.imshow('object detection',image_np)
        for i,b in enumerate(classes[0]):
            if classes[0][i] == 1 and scores[0][i]>=0.40:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                neck_loc=(boxes[0][i][0]+mid_y)/2
                head_loc=(boxes[0][i][0]+neck_loc)/2
                if SHOOT=="CENTER":
                    determine_movement(mid_x = mid_x, mid_y =mid_y,width=WIDTH,height=HEIGHT)
                elif SHOOT=="NECK":
                    determine_movement(mid_x = mid_x, mid_y =neck_loc,width=WIDTH,height=HEIGHT)
                elif SHOOT=="HEAD":
                    determine_movement(mid_x = mid_x, mid_y =head_loc,width=WIDTH,height=HEIGHT)
                else:
                    break
                    print("Something Wrong!")
                if INPUT=="keyboard":
                    keys.directKey(FIRE_KEY)
                    time.sleep(HL_TO_FIRE)
                    keys.directKey(FIRE_KEY,keys.key_release)
                    break
                if INPUT=="mouse":
                    keys.directMouse(0,0,FIRE_KEY)
                    time.sleep(HL_TO_FIRE)
                    keys.directMouse(0,0,RELEASE_KEY)
                    break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break