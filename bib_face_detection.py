import numpy as np
import cv2
import os

# default parameters
confidence_threshold = 0.5
nms_threshold = 0.25

# load class labels
path = os.getcwd()
meta_dir = os.path.sep.join([path, "yolo_cfg"])
labels = open(os.path.sep.join([meta_dir, "obj.names"])).read().strip().split("\n")

# load YOLO weights and configuration file
cfg = os.path.sep.join([meta_dir, "yolo-obj.cfg"])
weight = os.path.sep.join([path, "weights_backup/yolo-obj_4000.weights"])
# load YOLO detector trained on custom dataset
net = cv2.dnn.readNetFromDarknet(cfg, weight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine the output layer names
l_names = net.getLayerNames()
ol_names = [l_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

# load the image
image_path = os.path.sep.join([path, "images", "orig", "valid_tiny_50files", "1024_621C72DC-115C-8FFF-DF8B-66F8C302C598.JPG"])
image = cv2.imread(image_path)
if image is not None:
  (H,W) = image.shape[:2]
  # construct a blob from the input image, pass to the YOLO detector and
  # grab the bounding boxes and associated probabilities
  blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True, crop=False)
  net.setInput(blob)
  import time
  start = time.time()
  layer_outputs = net.forward(ol_names)
  end = time.time()
  print("Time: {:.6f}".format(end-start))

  # initialize some output lists
  boxes = []
  confidences = []
  classIDs = []

  # output of YOLO [0:4]: [center_x, center_y, box_w, box_h]
  # output of YOLO [4]: confidence
  # output of YOLO [5:]: class scores
  for output in layer_outputs:
      for detection in output:
          scores = detection[5:]
          classID = np.argmax(scores)
          confidence = scores[classID]

          if confidence > confidence_threshold:
              (center_x, center_y, width, height) = (detection[0:4] * ([W, H, W, H])).astype("int")
              x = int(center_x - (width / 2))
              y = int(center_y - (height / 2))
              boxes.append([x, y, int(width), int(height)])
              confidences.append(float(confidence))
              classIDs.append(classID)

      # perform Non-Maximum Suppression
      idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

  # fancy: initialize a list of colors to represent each possible class label
  np.random.seed(42)
  COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

  if len(idxs) > 0:
      for i in idxs.flatten():
          (x, y) = (boxes[i][0], boxes[i][1])
          (w, h) = (boxes[i][2], boxes[i][3])
          color = [int(c) for c in COLORS[classIDs[i]]]
          cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
          text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
          cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  #cv2_imshow(image)
  cv2.imshow("Image", image)
  cv2.waitKey(0)
else:
  print("No image is read.")