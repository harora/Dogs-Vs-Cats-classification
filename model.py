from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

ROWS = 64
COLS = 64
CHANNELS = 3

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

if __name__ == '__main__':
    model = network()
