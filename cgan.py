import numpy as np
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Convolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras.utils import np_utils
import keras.utils

def dataInit():
	print('Loading the data')
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


	X_train = np.concatenate((X_train, X_test), axis=0)
	X_train = (X_train.astype(np.float32) - 127.5)/127.5
	Y_train = np.concatenate((Y_train, Y_test), axis=0)

	Y_train_onehot = keras.utils.to_categorical(Y_train, 10)

	print('Training Data: ', X_train.shape)
	npRandom = np.random.RandomState(18)
	X_noise = []
	Y_noise = npRandom.randint(9, size=1000)

	for i in range(X_train.shape[0]):
		randomNoise = npRandom.uniform(-1,1,100)
		X_noise.append(randomNoise)
	X_noise = np.array(X_noise)
	Y_noise_onehot = keras.utils.to_categorical(Y_noise, 10)

	print('Random Noise Data: ', X_noise.shape)
	return X_train, Y_train_onehot, X_noise, Y_noise_onehot

def saveImage(imageData, imageName, epoch):
	f, ax = plt.subplots(16, 8)
	k = 0
	for i in range(16):
		for j in range(8):
			pltImage = imageData[k][0]
			ax[i,j].imshow(pltImage, interpolation='nearest',cmap='gray_r')
			ax[i,j].axis('off')
			k = k+1
	f.set_size_inches(18.5, 10.5)
	f.savefig('images/'+imageName+'_after_'+str(epoch)+'_epoch.png', dpi = 100, bbox_inches='tight', pad_inches = 0)
	plt.close(f)
	return None

if __name__ == '__main__':

	batchSize = 128
	nbEpoch = 200
	decayIter = 100
	lr = 0.0002

	X_train, Y_train, X_noise, Y_noise = dataInit()
	raise SystemExit
	X_train = X_train[:, np.newaxis, :, :]
	numExamples = (X_train.shape)[0]
	numBatches = int(numExamples/float(batchSize))

	print('Number of examples: ', numExamples)
	print('Number of Batches: ', numBatches)
	print('Number of epochs: ', nbEpoch)

	adam=Adam(lr=lr, beta_1=0.5 )

	print('Generator Model')

	generator = Sequential()
	g_noise = Input(shape=(100,))
	g_label = Input(shape=(10,))
	g_label_2 = Dense(units=50, activation='relu', kernel_initializer='glorot_normal')(g_label)
	g_input = concatenate(g_noise, g_label_2)
	generator.add(Dense(units=(128*7*7))(g_input)) #initialization
	generator.add(Activation('relu'))
	generator.add(Reshape((128, 7, 7)))
	generator.add(UpSampling2D(size=(2, 2)))
	generator.add(Conv2D(64, (5, 5), padding="same"))
	generator.add(Activation('relu'))		
	generator.add(UpSampling2D(size=(2, 2)))
	generator.add(Conv2D(1, (5, 5), padding='same'))
	generator.add(Activation('tanh'))
	
	print('Discriminator Model')

	discriminator = Sequential()
	discriminator.add(Conv2D(64, (5, 5), padding='same', strides=(2,2), input_shape=(1,28,28))) #initialization 
	discriminator.add(LeakyReLU(0.2))
	discriminator.add(Conv2D(128, (5, 5), padding='same', strides=(2,2)))
	discriminator.add(LeakyReLU(0.2))
	discriminator.add(Flatten())
	discriminator.add(Dense(1))
	discriminator.add(Activation('sigmoid'))
	discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

	discriminator.trainable = False

	print('DCGAN model')

	dcganInput = Input(shape=(100,))
	x = generator(dcganInput)
	dcganOutput = discriminator(x)
	dcgan = Model(inputs=dcganInput, outputs=dcganOutput)
	dcgan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

	discriminator.trainable = True

	if not os.path.exists('images'):
		os.makedirs('images')
	if not os.path.exists('models'):
		os.makedirs('models')
	if not os.path.exists('metrics'):
		os.makedirs('metrics')

	dLoss = []
	gLoss = []

	for epoch in range(1, nbEpoch + 1):
		print('Epoch: ', epoch)

		for i in range(numBatches):
			noisePredictBatch = X_noise[np.random.randint(numExamples, size = batchSize)]
			noiseDataBatch = generator.predict(noisePredictBatch)

			origDataBatch = X_train[np.random.randint(numExamples, size = batchSize)]
			noiseLabelsBatch, origLabelsBatch = np.zeros(batchSize).astype(int), np.ones(batchSize).astype(int)

			trainBatch = np.concatenate((noiseDataBatch, origDataBatch), axis = 0)
			trainLabels = np.concatenate((noiseLabelsBatch, origLabelsBatch))
			trainBatch, trainLabels = shuffle(trainBatch, trainLabels)

			discriminatorLoss = discriminator.train_on_batch(trainBatch, trainLabels)
			dcganLabels = np.ones(batchSize).astype(int)
			discriminator.trainable = False
			dcganLoss = dcgan.train_on_batch(noisePredictBatch, dcganLabels)
			discriminator.trainable = True

		dLoss.append(discriminatorLoss)
		gLoss.append(dcganLoss)
		
		if (epoch % 5 == 0) or (epoch == 1):
			saveImage(noiseDataBatch, 'generated', epoch)
			print('after epoch: ', epoch)
			print ('dcgan Loss: ', dcganLoss, '\t discriminator loss', discriminatorLoss)
			generator.save('models/generator_'+str(epoch)+'.h5')

		if epoch > decayIter :
			lrD = discriminator.optimizer.lr.get_value()
			lrG = generator.optimizer.lr.get_value()
			discriminator.optimizer.lr.set_value((lrD - lr/decayIter).astype(np.float32))
			generator.optimizer.lr.set_value((lrG - lr/decayIter).astype(np.float32))
			print('learning rate linearly decayed')

	np.save('metrics/dLoss.npy', np.array(dLoss))
	np.save('metrics/gLoss.npy', np.array(gLoss))
	print('Peace')
