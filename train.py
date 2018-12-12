#Sidharth Makhija (ssm9575@rit.edu) - 2018

import pdb
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.metrics import mean_absolute_error
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Dropout, Flatten, Merge
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from datagen_utils import createtrainGenerator,createtestGenerator
from keras.callbacks import ModelCheckpoint,TensorBoard,ReduceLROnPlateau


#PATHS - Update before running
main_path= 'Bone_Age_Data/'
exp_dir = 'Bone_Age/checkpoints/mobilenet_test/' 

#Load train and test image and gender feature stacks. Image size = n_sample x 224 x 224 x 3
train_label = np.load('Bone_Age/new_split/train_label_final.npy')
val_label = np.load('Bone_Age/new_split/val_labels_final.npy')
gender_feats_train = np.load('Bone_Age/new_split/train_genders_num.npy')
gender_feats_val = np.load('Bone_Age/new_split/val_genders_num.npy')
x_train = np.load('Bone_Age/new_split/x_train224_f.npy')
x_test = np.load('Bone_Age/new_split/x_val224.npy')
train_label = [int(i) for i in train_label]
val_label = [int(i) for i in val_label]
train_label = np.asarray(train_label)
val_label = np.asarray(val_label)
x_train = (1.0*x_train)/255
x_test = (1.0*x_test)/255

mobilenet_model = MobileNet(input_shape= (224, 224, 3), alpha=1.0, depth_multiplier=1, include_top=False, weights=None)
# model1 = InceptionV3(input_shape= (224, 224, 3), weights='imagenet', include_top = False)
for layer in mobilenet_model.layers:
    layer.trainable = True

temp_model = Sequential()
temp_model.add(mobilenet_model)
temp_model.add(Flatten())

model_gender = Sequential() 
model_gender.add(Dense(32, activation='relu',input_shape=(1,)))
model_gender.add(Dense(32, activation='relu'))

final_model = Sequential()
final_model.add(Merge([temp_model, model_gender], mode='concat')) #Merge image and gender feature vector
model.add(Dropout(0.3))
final_model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
final_model.add(Dense(512, activation='relu'))
final_model.add(Dense(1, activation = 'linear'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
final_model.compile(loss='mean_absolute_error', optimizer=adam,metrics=['mae','accuracy'])

#Save model checkpoints
filepath=exp_dir + "weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min',period = 20)
tensorboard = TensorBoard(log_dir=exp_dir + "logs/{}".format(time.time()))
reduce_lr = ReduceLROnPlateau(monitor='loss', mode='min', verbose=1,factor=0.5,patience=5, min_lr=0.0001, cooldown=5)
callbacks_list = [checkpoint,tensorboard,reduce_lr]

final_model.fit_generator(createtrainGenerator(x_train,gender_feats_train,train_label,10),steps_per_epoch=len(x_train) / 10, epochs=50,callbacks=callbacks_list)

#Save final model with weights
final_model.save('final_model.h5')
# model_json = final_model.to_json()
# with open("Nov_18_gen2_model_ep70.json", "w") as json_file:
    # json_file.write(model_json)
print ("Model and Weights Saved!!")

#Generate Test predictions
preds = final_model.predict_generator(createtestGenerator(x_test,gender_feats_val,1),steps=len(x_test) / 1,verbose=1)
np.save('preds_gen.npy',preds)
