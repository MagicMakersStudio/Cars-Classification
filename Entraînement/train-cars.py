
#-=-=-=-=-=-=-=-=-=-=-#
#       IMPORT        #
#-=-=-=-=-=-=-=-=-=-=-#

import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy
import h5py
from PIL import Image
from keras.optimizers import Adam
from glob import glob
from tqdm import tqdm


#-=-=-=-=-=-=-=-=-=-=-#
#        DATA         #
#-=-=-=-=-=-=-=-=-=-=-#

noms_dossiers=glob("../Data/dataset/downloads/*")

x_train = []
y_train = []
x_test = []
y_test = []
label = 0

for dossiers in tqdm(noms_dossiers):
	noms_image = glob(dossiers+"/*")
	for img_en_cours in tqdm(noms_image):
		try :
			mon_image = Image.open(img_en_cours)
			mon_image = mon_image.convert("RGB")
			mon_image = mon_image.resize((400,400))
			mon_tab = numpy.array(mon_image)
			x_train.append(mon_tab)
			y_train.append(label)
		except:
			pass
	label += 1

x_train, x_test, y_train, y_test=train_test_split(x_train, y_train, test_size=0.1)

x_train = numpy.array( x_train)
x_test = numpy.array( x_test)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

y_train = to_categorical(y_train,4)
y_test = to_categorical(y_test,4)


#-=-=-=-=-=-=-=-=-=-=-#
#       MODÈLE         #
#-=-=-=-=-=-=-=-=-=-=-#


model = Sequential()
model.add(Conv2D(12, kernel_size=5, strides=2, input_shape=(400, 400, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(24, kernel_size=5, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(32, kernel_size=5, strides=2, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(100, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4, activation="softmax"))

model.compile(loss = "categorical_crossentropy", optimizer=Adam(), metrics = ["accuracy"])

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#       TRAIN - ENTRAÎNEMENT        #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

model.fit(x_train, y_train, batch_size = 250, epochs = 300, validation_data = (x_test, y_test))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#       SAUVEGARDER MODÈLE        #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

model.save("car.h5")
