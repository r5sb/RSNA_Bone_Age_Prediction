#Sidharth Makhija (ssm9575@rit.edu) - 2018


import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def createtrainGenerator( X, I, Y,batch_size):

    while True:
        idx = np.random.permutation(X.shape[0])
        train_datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
        batches = train_datagen.flow( X[idx], Y[idx], batch_size=batch_size, shuffle=False)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]
            yield [batch[0], I[ idx[ idx0:idx1 ] ]], batch[1]
            idx0 = idx1
            if idx1 >= X.shape[0]:
                break
                
def createtestGenerator( X, I,batch_size):

    while True:
        idx = np.asarray(range(X.shape[0]))
        test_datagen = ImageDataGenerator()
        batches = test_datagen.flow( X[idx], batch_size=batch_size, shuffle=False)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch.shape[0]
            yield [batch, I[ idx[ idx0:idx1 ] ]]
            idx0 = idx1
            if idx1 >= X.shape[0]:
                break
