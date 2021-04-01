# import the necessary packages
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


# loading model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)
model = load_model("mask_detector.model")

# load the input image from disk=
image = cv2.imread("image.png")
orig = image.copy()
(h, w) = image.shape[:2]

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):

	confidence = detections[0, 0, i, 2]

	if confidence > args["confidence"]:

		# compute bounding box 
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# extract the face ROI
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# pass the face to the model
		(mask, withoutMask) = model.predict(face)[0]

		# predict
		if mask > withoutMask and mask > withoutProperly:
			label = "mask"
			color = (0, 255, 0)
		if withoutMask > mask and withoutMask > withoutProperly:
			label = "withoutmask"
			color = (0, 0, 255)
		if withoutProperly > withoutMask and withoutProperly > mask:
			label = "withoutproperly" 
			color = (255, 0, 0)


		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)