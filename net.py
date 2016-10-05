import os , random
import numpy as np
from PIL import Image

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


TRAIN_PATH = 'train/'
ROWS = 64
COLS = 64
CHANNELS = 3

def fetch_data():

		train_dogs =   [TRAIN_PATH+i for i in os.listdir(TRAIN_PATH) if 'dog' in i]
		train_cats =   [TRAIN_PATH+i for i in os.listdir(TRAIN_PATH) if 'cat' in i]
		train_files =  train_dogs[:1000] + train_cats[:1000]
		random.shuffle(train_files)

		return train_files

def generate_label(path):
	if 'dog' in path:
		label = 1
	if 'cat' in path:
		label = 0
	return label


def preprocess(path):

	image = np.array(Image.open(path))
	label = generate_label(path)
	# print image.shape
	image.resize((ROWS,COLS,CHANNELS))
	return image,label

def prepare_data(train_files):
	count = len(train_files)
	data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)
	labels = []
	for i, image_file in enumerate(train_files):
		image , label = preprocess(image_file)
		data[i] = image.T
		labels.append(label)
		if i%250 == 0: print('Processed {} of {}'.format(i, count))

	return data , labels

optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

def network():
	model = Sequential()
	model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(3,ROWS,COLS),activation='relu'))
	model.add(Convolution2D(32,3,3,border_mode='same',activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Convolution2D(64,3,3,border_mode = 'same',activation='relu'))
	model.add(Convolution2D(64,3,3,border_mode = 'same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(128,3,3,border_mode= 'same',activation = 'relu'))
	model.add(Convolution2D(128,3,3,border_mode= 'same',activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(256,3,3,border_mode='same',activation = 'relu'))
	model.add(Convolution2D(256,3,3,border_mode='same',activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	model.add(Dense(256,activation='relu'))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss= objective,optimizer= optimizer,metrics=['accuracy'])
	return model

def main():
	train_files = fetch_data()
	model = network()
	print "Model Compiled"

	train,labels = prepare_data(train_files)
	batch_size = 32
	nb_epoch=10
	for i in range(30):
		train_batch = train[(batch_size*i):(batch_size*(i+1))-1,:,:,:]
		label_batch = labels[batch_size*i:(batch_size*(i+1))-1]
		loss, acc = model.train_on_batch(train_batch, label_batch)
		print loss , acc



if __name__ == '__main__':
	main()
