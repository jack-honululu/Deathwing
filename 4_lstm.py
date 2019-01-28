import numpy as np
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.layers import (
    Input,
    Activation,
    LSTM,
    Dense,
    Flatten,
    concatenate
)
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
#parameters
batch_size = 100
hidden_units = 256
input_shape = (55,1)
classes = 5
nb_epoch = 100

X = np.load(('2_sequence_train_data.npy'))[:,:,None]
#X = X/np.max(X)
y = np.load('2_labels.npy')
'''
['DRI_Approach' 'DRI_Background' 'DRI_Challenge' 'DRI_Challenge_Goal'
 'DRI_Challenge_Hypothesis' 'DRI_FutureWork' 'DRI_Outcome'
 'DRI_Outcome_Contribution']
'''
#print(np.unique(y))
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-5)
#early_stopper = EarlyStopping(min_delta=0.001, patience=20)
csv_logger = CSVLogger('resnet50_base.csv')
#deal with y
y[y=='DRI_Approach'] = 0
y[y=='DRI_Background'] = 1
y[y=='DRI_Challenge'] = 2
y[y=='DRI_Challenge_Hypothesis'] = 2
y[y=='DRI_Challenge_Goal'] = 2
y[y=='DRI_FutureWork'] = 3
y[y=='DRI_Outcome'] = 4
y[y=='DRI_Outcome_Contribution'] = 4

y = np_utils.to_categorical(y, num_classes=5)

input = Input(shape = input_shape)
hidden = LSTM(55)(input)
outputs = Dense(units=classes,kernel_initializer="he_normal",activation="softmax")(hidden)
model = Model(inputs=input,outputs=outputs)

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])
model.summary()
def generator(features, labels, batch_size):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, features.shape[1],1))
    batch_labels = np.zeros((batch_size,5))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index= np.random.choice(len(features),1)
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels

print('training')
choose_index= np.random.randint(y.shape[0],size = 100)
total_index = np.arange(y.shape[0])
rest_index = np.array([x for x in total_index if x not in choose_index])
X_valid = X[choose_index]
y_valid = y[choose_index]
X_train = X[rest_index]
y_train = y[rest_index]

model.fit_generator(generator(X_train[:-100], y_train[:-100], batch_size),
                    steps_per_epoch = X_train[:-100].shape[0] / batch_size,
                    validation_data= generator(X_valid[-100:], y_valid[-100:],batch_size),
                    validation_steps = y_valid[-100:].shape[0] / batch_size,
                    epochs=nb_epoch, verbose=1, max_q_size=100,
                    callbacks=[lr_reducer, csv_logger])
print(choose_index)
