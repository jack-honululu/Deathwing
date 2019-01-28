import numpy as np
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from itertools import chain
#nltk.download('punkt') #first run with this then not needed anymore
#nltk.download('wordnet')
#nltk.download('stopwords')
#max sentence length 533
#max originwords length 55
data = np.load('1_data.npy')
print (data.shape)
X = data[:,0]
y = data[:,1]
print(np.unique(y))
i = 0
j =100
length = 0
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
tokenizer = Tokenizer(num_words=20000)
X1 = X[y!='DRI_Unspecified']
y = y[y!='DRI_Unspecified']
X2 = X1[y!='Sentence']
y = y[y!='Sentence']
tokenizer.fit_on_texts(X2)
argmaxx = 0
X3 = []
for x in X2:
    if (y[i] != 'Sentence') and (y[i] != 'DRI_Unspecified'):
        print(i)
        print(x,y[i])
        words = nltk.word_tokenize(x)
        no_stp_words = [word for word in words if not word in stop_words]
        origin_words = []
        for xx in no_stp_words:
            # print(lemmatizer.lemmatize('makes',wordnet.VERB))
            # print(lemmatizer.lemmatize('make',wordnet.NOUN))
            if len(xx)<2 and xx != 'R':
                continue
            #print('Vectorizing sequence data...')
            xx = lemmatizer.lemmatize(lemmatizer.lemmatize(xx, wordnet.VERB), wordnet.NOUN)
            if xx == 'discus':  #fix discuss bug
                xx = 'discuss'
            origin_words.append(xx)
        sequences = tokenizer.texts_to_sequences(origin_words)
        sequences = list(chain.from_iterable(sequences))
        #data = pad_sequences(np.array(sequences), value = 0,padding='post',maxlen=53)
        #data = pad_sequences(sequences, maxlen=50)
        #print(sequences)
        #print(data)
        #print(origin_words)
        if len(sequences)>length:
            argmaxx = i
            length = len(sequences)
        X3.append(sequences)
    i+=1
X3 = np.array(X3)
train_data = pad_sequences(X3,padding='post',maxlen=length)
print(length)
np.save('2_sequence_train_data',train_data)
np.save('2_labels',y)
