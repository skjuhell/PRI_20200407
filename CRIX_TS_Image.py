import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


import seaborn as sns
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten#create model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard

class Crypto():
	def __init__(self,path):
		self.data = pd.read_json(path).dropna()
		self.data.to_csv('crix.csv')
		

	def scale_data(self,a,b):
		self.a = a
		self.b = b
		self.minimum = np.min(self.data.price)
		self.maximum = np.max(self.data.price)

		self.data['price_sc'] = a+((self.data.price - self.minimum)*(b-a))/(self.maximum-self.minimum) 
		print('min value in df:'+str(np.min(self.data.price_sc)))
		print('max value in df:'+str(np.max(self.data.price_sc)))

	def ts_image_and_rp_plot(self,n_rows,n_cols,epsilon):
		self.n_rows = n_rows
		self.n_cols = n_cols
		self.seq_len = n_cols
		
		# 1 sequentialize the data
		seq_cont = []
		rec_mats = []
		y_images = []
		seqs = []
		for idx in range(self.data.price_sc.shape[0]-self.seq_len-1):
			
			ts = self.data.price_sc.values[idx:idx+self.seq_len]
			seq_cont.append(ts)
			
			if (idx+1>=self.seq_len):
				print(str(idx)+'/'+str(self.data.price_sc.shape[0]-self.seq_len-1))
				ts_matrix = np.array(seq_cont)[-self.seq_len:]
				
				# create a recurrence plot
				R = np.zeros((self.seq_len,self.seq_len))

				for x in range(self.seq_len):
					for y in range(self.seq_len):
						R[x,y] = 1 if np.abs(ts[x]-ts[y])<epsilon else 0

				# reset the seqs_container

				y_images.append(self.data.price_sc.values[idx+self.seq_len])
				seqs.append(ts_matrix)
				rec_mats.append(R)
				
				font = {'family': 'sans-serif',
						'color':  'red',
						'weight': 'bold',
						'size': 13,
						}

				
				'''
				gridsize = (2, 2)
				fig = plt.figure(figsize=(12, 10))
				ax1 = plt.subplot2grid(gridsize, (0, 0))
				ax2 = plt.subplot2grid(gridsize, (0, 1))
				ax3 = plt.subplot2grid(gridsize, (1, 0), colspan=2, rowspan=2)
				
				#Plot1
				sns.heatmap(ts_matrix,ax=ax1,cmap='Blues')
				ax1.set_title('TS Image',fontdict=font)
				ax1.invert_yaxis()
				
				#Plot2 
				sns.heatmap(R,ax=ax2,cmap='Blues',vmin=0,vmax=1)
				
				ax2.set_xticks(np.arange(0,n_rows,5).tolist())
				ax2.set_yticks(np.arange(0,n_rows,5).tolist())
				ax2.set_title('Recurrence Plot with '+chr(949)+'=0.0005',fontdict=font)
				ax2.invert_yaxis()

				#Plot3
				ax3.plot(self.data['date'][np.arange(idx,idx+self.seq_len)],ts)
				# Hide the right and top spines
				ax3.spines['right'].set_visible(False)
				ax3.spines['top'].set_visible(False)

				# Only show ticks on the left and bottom spines
				ax3.yaxis.set_ticks_position('left')
				ax3.xaxis.set_ticks_position('bottom')

				ax3.set_title('CRIX',fontdict=font)
				plt.tight_layout()
				plt.savefig('crix_ts_rp_e'+str(epsilon).replace('.','_')+'/CRIX_'+str(idx)+'.png',transparent=True)
				plt.close()
				'''
				
				


		self.seqs = seqs
		np.save('seqs_and_rp.npy',seqs)
		self.y_images = y_images
		np.save('y_values.npy',y_images)
		self.rec_mats = rec_mats
		np.save('rp_matrix.npy',rec_mats)
				

	def train_test_split(self,ratio):

		#merge the RP plots and the time series images
		#therefore add a dimension
		self.seqs,self.recs = map(lambda x: np.expand_dims(x,axis=3),[self.seqs,self.rec_mats] )
		self.ts_images = np.concatenate([self.seqs,self.recs],axis=3)
		self.y_images = np.array(self.y_images)

		#split the dataframe
		split = int(np.round(self.ts_images.shape[0]*ratio,0))
		self.train = np.arange(0,split)


		self.test = np.arange(split,self.ts_images.shape[0])

		self.ts_images_train = self.ts_images[self.train]
		self.y_images_train = self.y_images[self.train]

		self.ts_images_test = self.ts_images[self.test]
		self.y_images_test = self.y_images[self.test]

		np.save('crix_train_data_X.npy',self.ts_images_train)
		np.save('crix_train_data_Y.npy',self.y_images_train)

		np.save('crix_test_data_X.npy',self.ts_images_test)
		np.save('crix_test_data_Y.npy',self.y_images_test)


ts_rows = 10
ts_cols = 10


crix = Crypto('crix.json')
crix.scale_data(-1,1)
crix.ts_image_and_rp_plot(n_rows=ts_rows,n_cols=ts_cols,epsilon=0.0005)

crix.train_test_split(0.9)

logdir = "logs/run" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)
#grads_and_vars = optimizer.compute_gradients(tf_loss)


model = Sequential()#add model layers
model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(ts_rows,ts_cols,2)))
#model.add(BatchNormalization())
model.add(Conv2D(8, kernel_size=3, activation='relu'))
#model.add(BatchNormalization())
model.add(Conv2D(4, kernel_size=3, activation='tanh'))
model.add(Flatten())
model.add(Dense(1))


checkpointer = ModelCheckpoint(filepath="best_weights_crix_cnn.hdf5", 
							   monitor = 'val_loss',
							   verbose=1, 
							   save_best_only=True)

model.compile(optimizer='adam', loss='mse',metrics=['mae'])
history = model.fit(crix.ts_images_train, crix.y_images_train,
					callbacks=[checkpointer,tensorboard_callback],
					epochs=40,
					batch_size=8,
					validation_data=[crix.ts_images_test,crix.y_images_test])

mae = history.history['mean_absolute_error']
val_mae = history.history['val_mean_absolute_error']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(mae) + 1)

fig, axs = plt.subplots(2)
axs[0].plot(epochs, mae, 'bo', label='Training mae')
axs[0].plot(epochs, val_mae, 'b', label='Validation mae')
axs[0].set_title('Training and validation MAE')
axs[0].legend()
axs[1].plot(epochs, loss, 'bo', label='Training loss')
axs[1].plot(epochs, val_loss, 'b', label='Validation loss')
axs[1].set_title('Training and validation loss')
axs[1].legend()

plt.tight_layout()
fig.savefig('Train_Test_Performance.png')
plt.close('all')


model = load_model('best_weights_crix_cnn.hdf5')
preds = model.predict(crix.ts_images_test)

################################################################################
#################### Plot Results ##############################################
################################################################################
font = {'family': 'sans-serif',
		'color':  'red',
		'weight': 'bold',
		'size': 13,
		}

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(crix.data.date[crix.test],preds, label="CRIX Prediction",color='red')
ax.plot(crix.data.date[crix.test],crix.y_images_test, label="CRIX Price",color='royalblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          frameon=False, ncol=5)


ax.set_title('CRIX test set predictions',fontdict=font)

plt.savefig('CRIX_pred.png',transparent=True)


