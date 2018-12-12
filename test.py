#Sidharth Makhija (ssm9575@rit.edu) - 2018

import pdb
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import load_model
from keras.models import model_from_json
from sklearn.metrics import mean_absolute_error
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.generic_utils import CustomObjectScope
from datagen_utils import createtrainGenerator,createtestGenerator


model_wts = '/model_weights.hdf5'
x_test = np.load('/Bone_Age/new_split/x_val224.npy')
val_label = np.load('/Bone_Age/new_split/val_labels_final.npy')
gender_feats_val = np.load('Bone_Age/new_split/val_genders_num.npy')

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights(model_wts)
# print("Loaded model from disk")

with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    loaded_model = load_model(model_wts)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
loaded_model.compile(loss='mean_absolute_error', optimizer=adam,metrics=['mae'])#'mean_squared_error'

#Generate Test predictions
preds = loaded_model.predict_generator(createtestGenerator(x_test,gender_feats_val,1),steps=len(x_test) / 1,verbose=1)
np.save('preds_gen.npy',preds)

preds = loaded_model.predict_on_batch(x_test)
np.save('/Bone_Age/preds.npy',preds)
mae = mean_absolute_error(val_label, preds)
print ("MAE2 = {}".format(mae))

#Plot results
# fig, ax1 = plt.subplots(1,1, figsize = (8,8))
# ax1.plot(val_label, preds2, 'r.', label = 'predictions MobileNet')
# ax1.plot(val_label, val_label, 'b-', label = 'actual')
# ax1.legend()
# ax1.set_xlabel('Actual Age (Months)')
# ax1.set_ylabel('Predicted Age (Months)')
# plt.save('results.png')




