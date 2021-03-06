#!/usr/bin/env python
# coding: utf-8

# # Import packages

# In[4]:


import os
import numpy as np
import pandas as pd
import jieba
from tensorflow.keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from tensorflow.keras.losses import categorical_crossentropy
from hanziconv import HanziConv
from keras.layers import *
os.chdir('C:/Users/TimHsu/Documents/Cloud/DeepLearning/final_kaggle')


# In[5]:


os.getcwd()


# # Input data

# In[6]:


train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')


# #### load label, title and keyword 

# In[11]:


train_data_label = train_data['label_name']
train_data_text = train_data['title']
train_data_key = train_data['keyword']
# train_data_text = []
# for line in train_data_text_simple:
#     train_data_text.append(HanziConv.toTraditional(line)) # 簡轉繁
test_data_text = test_data['title']
test_data_key = test_data['keyword']
# test_data_text = []
# for line in test_data_text_simple:
#     test_data_text.append(HanziConv.toTraditional(line)) # 簡轉繁
text_full = pd.concat([train_data_text, test_data_text])
key_full = pd.concat([train_data_key, test_data_key])
text_full = text_full.reset_index(drop=True)
key_full = key_full.reset_index(drop=True)


# In[10]:


key_full


# #### split key word

# In[12]:


#先把keyword以逗點的形式切好，等等再丟到辭庫裡面
key_full = key_full.dropna(',')  
#辭庫用的list之後就用不到了
key_split_ = []
for i in key_full.index:
    key_split_ = key_split_ + (key_full[i].split(','))
file = open('keyword_list.txt','w', encoding="utf-8")
for i in key_split_:
    file.writelines([i+"\n"])
file.close()


# In[15]:





# In[4]:


words_skip = [line.strip() for line in open('tt.txt',encoding='UTF-8').readlines()]
# dict
jieba.load_userdict('keyword_list.txt')
# 切資料用的function
def data_to_vector(data):
    data_split = list(jieba.cut(data))
    outcome = []
    for i in data_split:
        if !(i in words_skip):
            outcome.append(i)
    return(outcome)


# In[12]:


key_full_w_na = key_full.fillna(',')
text_full_str = text_full.astype('str')
key_full_w_na_str = key_full_w_na.astype('str')
corpus_text= text_full_str.apply(data_to_vector)
corpus_key = key_full_w_na_str.apply(data_to_vector)
#key 跟 text合在一起
corpus_text_key = corpus_text + corpus_key


# In[13]:





# In[57]:


corpus_text_key[0:2]


# # Preprocess model in Keras

# In[14]:


import keras
MAX_NUM_WORDS = 10000
tokenizer = keras     .preprocessing     .text     .Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(corpus_text_key)


# In[15]:


X = tokenizer.texts_to_sequences(corpus_text_key)


# In[16]:


type(corpus_text_key)


# In[ ]:


X[0:2]


# In[49]:


#0 padding
MAX_SEQUENCE_LENGTH = 30
X = keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y_train = keras.utils.to_categorical(train_data['label'])


# In[51]:


optimizer = Adam()
NUM_CLASSES = 10
NUM_EPOCHS = 3
BATCH_SIZE = 64
NUM_EMBEDDING_DIM = 64
NUM_LSTM_UNITS = 32
verbosity = 1
from keras import Input
from keras.layers import Embedding, LSTM, concatenate, Dense
from keras.models import Model
top_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
embedding_layer = Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
top_embedded = embedding_layer(top_input)
shared_lstm = LSTM(NUM_LSTM_UNITS)
top_output = shared_lstm(top_embedded)
dense =  Dense(units=NUM_CLASSES, activation='softmax')
predictions = dense(top_output)
model = Model(inputs=top_input,  outputs=predictions)
model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])


# In[ ]:


history = model.fit(X[0:len(train_data)], keras.utils.to_categorical(train_data['label']),
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          verbose=verbosity,
          shuffle=True)


# In[52]:


pred  = model.predict(X[len(train_data):len(X)])


# In[54]:


pred = pred.argmax(axis=1)


# In[ ]:


pred


# In[55]:


pred.to_csv('submission.csv', index=None)


# In[46]:


type(X_train)


from keras import backend as K

arr = np.array([1.0, 2.0], dtype='float64')

new_arr = K.cast_to_floatx(arr)
type(new_arr)

