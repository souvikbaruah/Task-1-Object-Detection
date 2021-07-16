# cv2 ia a open source python computer vision library
import cv2
# Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
import matplotlib.pyplot as plt

# importing ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
# It's a pre trained model for identifying objects from an image or video

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# frozen_inference_graph.pb, is a frozen graph that cannot be trained anymore,
# it defines the graphdef and is actually a serialized graph and can be loaded with this code

frozen_model = 'frozen_inference_graph.pb'  # importing

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = [0]  # empty list of python

file_name = 'coco.names'

with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    classLabels.append(fpt.read())

print(classLabels)
print(len(classLabels))

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)  # 255/2 = 127.5
model.setInputMean((127.5, 127.5, 127.5))  # mobilenet => [-1,1]
model.setInputSwapRB(True)

#                         here we are inserting pic.jpg image

img = cv2.imread('pic.jpg')

plt.imshow(img)

cv2.imshow("Image", img)

classIndex, confidece, bbox = model.detect(img, confThreshold=0.5)
print(classIndex)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for classInd, conf, boxes in zip(classIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
                color=(0, 255, 0), thickness=3)

# plt.imshow for to plot identified objects in image

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# For to display objects detected in inserted image
# It displays the ploted image

cv2.imshow("Image", img)

#
#
#                          For to detect objects in Video

cap = cv2.VideoCapture("video.mp4")  # importing "video.mp4"

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise print("cannot open video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    print(ClassIndex)
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 255, 0), thickness=3)
    cv2.imshow('Object Detection ', frame)

    if cv2.waitKey(2) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
