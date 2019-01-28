import numpy as np
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.layers.convolutional import (
    Conv1D,
    MaxPooling1D,
    AveragePooling1D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
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
type = 50
batch_size = 100
hidden_units = 256
input_shape = (55,1)
num_outputs = 5
nb_epoch = 500
#repetitions = [2, 2, 2, 2] #res18
repetitions = [3, 4, 6, 3]

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
#early_stopper = EarlyStopping(min_delta=0.001, patience=30)
csv_logger = CSVLogger('resnet50_second.csv')
checkpoint  = ModelCheckpoint('./resnet50_weights_file.h5', monitor='val_acc',
                              verbose=0, save_best_only=True,
                            save_weights_only=True, mode='max', period=1)
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
#model build-in
def _bn_relu(input):
    norm = BatchNormalization(axis=2)(input)
    return Activation("relu")(norm)

def _bn_relu_conv(input,filters, kernel_size,
                      strides=1, padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=l2(1.e-4)):

    activation = _bn_relu(input)
    return Conv1D(filters=filters, kernel_size=kernel_size,
                  strides=strides, padding=padding,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer)(activation)

def _shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1]/residual_shape[1]))
    equal_channels = input_shape[2] == residual_shape[2]
    shortcut = input
    # 1 conv if shape is different. Else identity.
    if stride_width > 1 or not equal_channels:
        shortcut = Conv1D(filters=residual_shape[2],
                          kernel_size=1,
                          strides=stride_width,
                          padding="same",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(type,input,filters, repetitions, is_first_layer=False):
    if type == 50 or type == 101 or type == 152:
        blocktype = bottleneck
    if type == 34 or type == 18:
        blocktype = basic_block
    for i in range(repetitions):
        init_strides = 1
        if i == 0 and not is_first_layer:
            init_strides = 2
        input = blocktype(input,filters=filters, init_strides=init_strides,is_first_block_of_first_layer=(is_first_layer and i == 0))
    return input


def bottleneck(input,filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """
    Return a final conv layer of filters * 4
    """
    if is_first_block_of_first_layer:
        # don't repeat bn->relu since we just did bn->relu->maxpool
        conv_1_1 = Conv1D(filters=filters, kernel_size=1,
                          strides=init_strides,
                          padding="same",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(1e-4))(input)
    else:
        conv_1_1 = _bn_relu_conv(input,filters=filters, kernel_size=1,
                                 strides=init_strides)

    conv_3_3 = _bn_relu_conv(conv_1_1,filters=filters, kernel_size=3)
    residual = _bn_relu_conv(conv_3_3,filters=filters * 4, kernel_size=1)
    return _shortcut(input, residual)

def basic_block(input, filters, init_strides=1, is_first_block_of_first_layer=False):
    if is_first_block_of_first_layer:
        # don't repeat bn->relu since we just did bn->relu->maxpool
        conv1 = Conv1D(filters=filters, kernel_size=3,
                       strides=init_strides,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4))(input)
    else:
        conv1 = _bn_relu_conv(input,filters=filters, kernel_size=3,
                              strides=init_strides)

    residual = _bn_relu_conv(input,filters=filters, kernel_size=3)
    return _shortcut(input, residual)

def Resnet(type,input_shape, num_outputs, repetitions):
    input = Input(shape=input_shape)
    conv1 = Conv1D(filters=64,kernel_size = 7,strides=2)(input)
    conv1 = _bn_relu(conv1)
    #print(conv1.shape)
    pool1 = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv1)
    block = pool1
    #print('first',block.shape)
    filters = 64
    for i, r in enumerate(repetitions):
        block = _residual_block(type,block,filters=filters, repetitions=r, is_first_layer=(i == 0))
        #print(block.shape)
        filters *= 2

    # Last activation
    block = _bn_relu(block)

    # Classifier block
    block_shape = K.int_shape(block)
    pool2 = AveragePooling1D(pool_size=block_shape[1],
                             strides=1)(block)
    flatten1 = Flatten()(pool2)
    dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                  activation="softmax")(flatten1)

    return input,dense

#compile
print('startmodeling')
input,dense = Resnet(type,input_shape=input_shape, num_outputs = num_outputs, repetitions = repetitions)
model = Model(inputs = input,outputs = dense)
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

model.fit_generator(generator(X_train, y_train, batch_size),
                    steps_per_epoch = X_train.shape[0] / batch_size,
                    validation_data= generator(X_valid, y_valid,batch_size),
                    validation_steps = y_valid.shape[0] / batch_size,
                    epochs=nb_epoch, verbose=1, max_q_size=100,
                    callbacks=[lr_reducer, csv_logger,checkpoint])
print(choose_index)
