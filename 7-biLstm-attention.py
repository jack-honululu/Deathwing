from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, Dropout
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
import numpy as np
#parameter
maxlen = 55
max_features = 8299
embed_size = 32
batch_size = 100
nb_epoch = 100


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim



def BidLstm(maxlen, max_features, embed_size):#, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size,input_length = maxlen)(inp)#, weights=[embedding_matrix],
                  #trainable=False)(inp)
    x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25,
                           recurrent_dropout=0.25))(x)
    x = Attention(maxlen)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(5, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model

X_origin = np.load(('2_sequence_train_data.npy'))[:,:,None]
y = np.load('2_labels.npy')
X = X_origin.reshape(X_origin.shape[0],X_origin.shape[1])#padded ints matrix
#print(np.unique(y))
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-5)
#early_stopper = EarlyStopping(min_delta=0.001, patience=30)
csv_logger = CSVLogger('bilstm_attention_1st.csv')
filepath="save_models/bilstm_attention-models-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint  = ModelCheckpoint(filepath, monitor='val_acc',
                              verbose=0, save_best_only=False,
                            save_weights_only=False, mode='max', period=1)
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

#modeling
print('startmodeling')
model = BidLstm(maxlen, max_features, embed_size)

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])
model.summary()
def generator(features, labels, batch_size):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, features.shape[1]))
    batch_labels = np.zeros((batch_size,5))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index= np.random.choice(len(features),1)
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels

print('training')
choose_index= np.array([839,316 ,4045, 3113, 4167, 3036, 3385, 7329, 3696, 1518, 4131, 4651, 5935, 6183, 7156 ,7620,  812, 6863, 1221, 7075, 4136, 7198, 8095, 1097, 1268, 8149, 2839, 2667, 5534, 3843, 6782, 6322, 1416, 2621, 3298, 6243, 7607, 1776, 5789, 4993, 8510,  303, 5961, 3575, 7045, 1656, 5121,  733, 5291, 1260,  606, 6645, 1208, 2251, 4757,  363,7473, 3535, 2841, 3431,  903, 4187, 8366, 1723, 8250, 4869, 1887, 1614, 3413, 5489,7794, 6649, 7529, 2732, 4536, 7345, 1784, 1736,  505, 3813, 6121, 4572, 5772, 1290,2281, 8481, 5110, 7789, 6511, 3135,  367, 6637, 6988, 3057, 6980,  837, 2887, 4137,5659, 1105])
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
'''
[5037 4309 7002 2679 3867 1951 4702 6218 3204 6290 4538 1104 8021 2413
 3777 8353  468  382 2373 3632 6298 2747 2060 1118 2950  825  504 1769
 5773 5542 5406 2341 6897 7996 8319 4388 4129  265 4277 5840 6870 1686
 1255 6252 4767 2723 6095 2607 7378 3489 8320 5156  107 5154 5256  837
 6033 1263 7301 6779 6057 8048 2851 4290 3631  932 7716 3696 6413 3799
 7515  783 6724 6314 1135 1832 6070  546 1586 3422 6096  361  480 8059
 3248 8108 4013 5472 7123  222 5533 6101 6786 7741 1578 1964  124 3614
 5033 6397]
'''
