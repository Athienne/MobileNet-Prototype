from msilib.schema import Class
import cv2
from cv2 import imshow
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

# LOAD MODEL
model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# READ IMAGE
img = cv2.imread('people_cctv.png')

# CONFIGURATION FOR DETECTING OBJECTES
ClassIndex, confidence, bbox = model.detect(img, confThreshold = 0.5)
font_scale = 1
font = cv2.FONT_HERSHEY_COMPLEX
people_count = 0

# DRAW BOXES ON DETECTED OBJECTS
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    if ClassInd == 1:
        cv2.rectangle(img, boxes, (255, 0, 0), 2)
        cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
        people_count += 1

print("People Count:", people_count)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.waitforbuttonpress()
