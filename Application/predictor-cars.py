
#-=-=-=-=-=-=-=-=-=-=-#
#       IMPORT        #
#-=-=-=-=-=-=-=-=-=-=-#

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy
import h5py
from PIL import Image
from glob import glob
from tqdm import tqdm

#-=-=-=-=-=-=-=-=-=-=-#
#   CHARGER MODÈLE    #
#-=-=-=-=-=-=-=-=-=-=-#

model=load_model("car.h5")


#-=-=-=-=-=-=-=-=-=-=-#
#        DATA         #
#-=-=-=-=-=-=-=-=-=-=-#

noms_dossiers=glob(".../Data/test/*")

#-=-=-=-=-=-=-=-=-=-=-#
#     PREDICTION      #
#-=-=-=-=-=-=-=-=-=-=-#

for img_en_cours in tqdm(noms_dossiers):
	try :
		mon_image = Image.open(img_en_cours)
		mon_image = mon_image.convert("RGB")
		mon_image = mon_image.resize((400,400))
		mon_tab = numpy.array(mon_image)
		mon_tab = mon_tab.astype("float32")
		mon_tab /= 255
		mon_tab= mon_tab.reshape(1,400,400,3)
		prediction = model.predict(mon_tab)
		prediction = numpy.argmax(prediction)
		print(img_en_cours + "------------>" + str(prediction))
	except:
		pass
