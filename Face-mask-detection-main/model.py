# basic libraries 
import matplotlib.pyplot as plt
import numpy as np
import os

# tensorflow libraries for model building
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

# other libraries for pre processing
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

DIRECTORY = r"C:\Users\Kausthub Thekke Mada\Desktop\face-mask-detector\dataset"
CATEGORIES = ["with_mask", "without_mask", "without_properly"]

# loading images/dataset
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# perform one-hot encoding on the labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, stratify=labels)

# data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# head model 
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False


# compiling, fitting and making predictions respectively
opt = Adam(lr=1e-4, decay=1e-4 / 50)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
Hello = model.fit( aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), epochs=50)
predIdxs = model.predict(testX, batch_size=32)
predIdxs = np.argmax(predIdxs, axis=1)

# classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=le.classes_))

# plot the graph

x = np.arange(0, 50)
plt.style.use("ggplot")
plt.figure()
plt.plot(x, Hello.history["loss"], label="train_loss")
plt.plot(x, Hello.history["val_loss"], label="val_loss")
plt.plot(x, Hello.history["accuracy"], label="train_acc")
plt.plot(x, Hello.history["val_accuracy"], label="val_acc")
plt.title("Loss and Accuracy")
plt.xlabel("Epoch no.")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

# saving the model
model.save("mask_detector.model", save_format="h5")