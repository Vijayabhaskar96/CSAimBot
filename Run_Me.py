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
import argparse
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

keystochose=r'"1","2","3","4","5","6","7","8","9","0","NUMPAD1","NUMPAD2","NUMPAD3","NUMPAD4","NUMPAD5","NUMPAD6","NUMPAD7","NUMPAD8","NUMPAD9","NUMPAD0","DIVIDE","MULTIPLY","SUBSTRACT","ADD","DECIMAL","NUMPADENTER","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","UP","LEFT","RIGHT","DOWN","ESC","SPACE","RETURN","INSERT","DELETE","HOME","END","PRIOR","NEXT","BACK","TAB","LCONTROL","RCONTROL","LSHIFT","RSHIFT","LMENU","RMENU","LWIN","RWIN","APPS","CAPITAL","NUMLOCK","SCROLL","MINUS","LBRACKET","RBRACKET","SEMICOLON","APOSTROPHE","GRAVE","BACKSLASH","COMMA","PERIOD","SLASH"'
parser=argparse.ArgumentParser()
parser.add_argument('--width',type=int,default=800,help="Width of the game resolution(default:800)")
parser.add_argument('--height',type=int,default=600,help="Height of the game resolution(default:600)")
parser.add_argument('--resize',type=int,default=4,help="Keep this as low as possible to get better detection of person but decreasing it also reduces the frame rate of what bot sees.(default:4)")
parser.add_argument('--score',type=float,default=0.40,help="Increase as long as the bot detects the person,Decrease if bot can't detect the person.(default:0.40)")
parser.add_argument('--show',type=bool,default=True,help="Set to False if you don't want to see the captured screen(default:True)")
parser.add_argument('--input',type=str,default="keyboard",help='(Enter Without Quotes)Choose between "keyboard" and "mouse".(default:keyboard).Choose the --key if chose keyboard)')
parser.add_argument('--key',type=str,default="RETURN",help="(Enter Without Quotes)Choose Anyone from".join(keystochose))
parser.add_argument('--shoot',type=str,default="CENTER",help='(Enter Without Quotes) Shoots at CENTER of the person detected by default(choose between:CENTER,HEAD,NECK)')
parser.add_argument('--duration',type=float,default=0.4,help='How long to shoot(in seconds),default:0.4 seconds')
args=parser.parse_args()

#****************** Parameters to choose ******************

#Resolution of the game(Make sure the game runs in windowed mode in the top left corner of your screen
WIDTH=args.width
HEIGHT=args.height
RESIZE_FACTOR=args.resize #Keep this as low as possible to get better detection of person but it also reduces the frame rate of what bot sees
SCORE=args.score  #Increase as long as the bot detects the person,Decrease if bot can't detect the person.
SHOW_CAPTURE=args.show #Set to false if you don't want to see the captured screen

#Fire using Keyboard or Mouse?
#HIGHLY RECOMMENDED TO USE A KEYBOARD KEY TO SHOOT
INPUT=str(args.input) #if you want set it to "mouse"

#WHICH KEY SHOULD BE PRESSED INORDER TO FIRE?
#HIGHLY RECOMMENDED TO USE A KEYBOARD KEY TO SHOOT
if INPUT=="keyboard":
    FIRE_KEY=str(args.key) #REFER The keys.py file for list of all keys
elif INPUT=="mouse":
    FIRE_KEY=keys.mouse_lb_press #If you want to change look keys.py file
    RELEASE_KEY=keys.mouse_lb_release
#Where to Shoot?
SHOOT=args.shoot #You can use "HEAD" or "NECK" or "CENTER"
# CENTER - Target at 50% of the height of the detected person
# NECK - Target at 25% of the height of the detected person
# HEAD - Target at 12.5% of the height of the detected person

#How long to press the fire button when a person is detected?
HL_TO_FIRE=args.duration #HOLDS THE FIRE BUTTON FOR 0.4 SECONDS

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
    print("Running with Parameters:\nDevice: {},WIDTH:{} HEIGHT:{} SHOOT:{} DURATION:{}".format(INPUT,WIDTH,HEIGHT,SHOOT,HL_TO_FIRE))
    while True:
        screen = cv2.resize(grab_screen(region=(0,0,WIDTH,HEIGHT)), (int(WIDTH/RESIZE_FACTOR),int(HEIGHT/RESIZE_FACTOR)))
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
        if SHOW_CAPTURE:
            cv2.imshow('Press q to quit bot',image_np)
        for i,b in enumerate(classes[0]):
            if classes[0][i] == 1 and scores[0][i]>=SCORE:
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
                    print("Something Wrong!")
                    break
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