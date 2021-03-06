'''
this script trains an LSTM model on one of the data files in the data folder of
this repository. the input file can be changed to another file from the data folder
by changing its name in line 46.

it is recommended to run this script on GPU, as recurrent networks are quite 
computationally intensive.

Author: Niek Tax
'''

from __future__ import print_function, division
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.utils.data_utils import get_file
from keras.optimizers import Nadam, Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from collections import Counter
import unicodecsv
import numpy as np
import random
import sys
import os
import copy
import csv
import time
from itertools import izip
from datetime import datetime
from math import log
import sys, os
if len(sys.argv)<2:
    sys.exit("python train.py eventlog n_neurons n_layers")
    
#eventlog = "helpdesk.csv"  
eventlog = sys.argv[1]      
n_neurons=int(sys.argv[2])
n_layers=int(sys.argv[3])
lastcase = ''
line = ''
firstLine = True
lines = []
timeseqs = []
timeseqs2 = []
times = []
times2 = []
numlines = 0
casestarttime = None
lasteventtime = None

csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
ascii_offset = 161

for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    if row[0]!=lastcase:
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:        
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
        line = ''
        times = []
        times2 = []
        numlines+=1
    line+=unichr(int(row[1])+ascii_offset)
    timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    times.append(timediff)
    times2.append(timediff2)
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
numlines+=1

divisor = np.mean([item for sublist in timeseqs for item in sublist])
print('divisor: {}'.format(divisor))
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))

elems_per_fold = int(round(numlines/3))
fold1 = lines[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]

fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2*elems_per_fold]

fold3 = lines[2*elems_per_fold:]
fold3_t = timeseqs[2*elems_per_fold:]
fold3_t2 = timeseqs2[2*elems_per_fold:]

lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2

step = 1
sentences = []
softness = 0
next_chars = []
lines = map(lambda x: x+'!',lines)
maxlen = max(map(lambda x: len(x),lines))

chars = map(lambda x : set(x),lines)
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars)
chars.remove('!')
print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
print(indices_char)


csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
lastcase = ''
line = ''
firstLine = True
lines = []
timeseqs = []
timeseqs2 = []
timeseqs3 = []
timeseqs4 = []
times = []
times2 = []
times3 = []
times4 = []
numlines = 0
casestarttime = None
lasteventtime = None
for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    if row[0]!=lastcase:
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:        
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
            timeseqs3.append(times3)
            timeseqs4.append(times4)
        line = ''
        times = []
        times2 = []
        times3 = []
        times4 = []
        numlines+=1
    line+=unichr(int(row[1])+ascii_offset)
    timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    timediff3 = timesincemidnight.seconds
    timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday()
    times.append(timediff)
    times2.append(timediff2)
    times3.append(timediff3)
    times4.append(timediff4)
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
timeseqs3.append(times3)
timeseqs4.append(times4)
numlines+=1

elems_per_fold = int(round(numlines/3))
fold1 = lines[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]
fold1_t3 = timeseqs3[:elems_per_fold]
fold1_t4 = timeseqs4[:elems_per_fold]
with open('output_files/folds/'+eventlog+'fold1.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in izip(fold1, fold1_t):    
        spamwriter.writerow([unicode(s).encode("utf-8") +'#{}'.format(t) for s, t in izip(row, timeseq)])

fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2*elems_per_fold]
fold2_t3 = timeseqs3[elems_per_fold:2*elems_per_fold]
fold2_t4 = timeseqs4[elems_per_fold:2*elems_per_fold]
with open('output_files/folds/'+eventlog+'fold2.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in izip(fold2, fold2_t):
        spamwriter.writerow([unicode(s).encode("utf-8") +'#{}'.format(t) for s, t in izip(row, timeseq)])
        
fold3 = lines[2*elems_per_fold:]
fold3_t = timeseqs[2*elems_per_fold:]
fold3_t2 = timeseqs2[2*elems_per_fold:]
fold3_t3 = timeseqs3[2*elems_per_fold:]
fold3_t4 = timeseqs4[2*elems_per_fold:]
with open('output_files/folds/'+eventlog+'fold3.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in izip(fold3, fold3_t):
        spamwriter.writerow([unicode(s).encode("utf-8") +'#{}'.format(t) for s, t in izip(row, timeseq)])

lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2
lines_t3 = fold1_t3 + fold2_t3
lines_t4 = fold1_t4 + fold2_t4

step = 1
sentences = []
softness = 0
next_chars = []
lines = map(lambda x: x+'!',lines)

sentences_t = []
sentences_t2 = []
sentences_t3 = []
sentences_t4 = []
next_chars_t = []
next_chars_t2 = []
next_chars_t3 = []
next_chars_t4 = []
for line, line_t, line_t2, line_t3, line_t4 in izip(lines, lines_t, lines_t2, lines_t3, lines_t4):
    for i in range(0, len(line), step):
        if i==0:
            continue
        sentences.append(line[0: i])
        sentences_t.append(line_t[0:i])
        sentences_t2.append(line_t2[0:i])
        sentences_t3.append(line_t3[0:i])
        sentences_t4.append(line_t4[0:i])
        next_chars.append(line[i])
        if i==len(line)-1: # special case to deal time of end character
            next_chars_t.append(0)
            next_chars_t2.append(0)
            next_chars_t3.append(0)
            next_chars_t4.append(0)
        else:
            next_chars_t.append(line_t[i])
            next_chars_t2.append(line_t2[i])
            next_chars_t3.append(line_t3[i])
            next_chars_t4.append(line_t4[i])
print('nb sequences:', len(sentences))

print('Vectorization...')
num_features = len(chars)+5
print('num features: {}'.format(num_features))
X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
y_t = np.zeros((len(sentences)), dtype=np.float32)
for i, sentence in enumerate(sentences):
    leftpad = maxlen-len(sentence)
    next_t = next_chars_t[i]
    sentence_t = sentences_t[i]
    sentence_t2 = sentences_t2[i]
    sentence_t3 = sentences_t3[i]
    sentence_t4 = sentences_t4[i]
    for t, char in enumerate(sentence):
        multiset_abstraction = Counter(sentence[:t+1])
        for c in chars:
            if c==char:
                X[i, t+leftpad, char_indices[c]] = 1
        X[i, t+leftpad, len(chars)] = t+1
        X[i, t+leftpad, len(chars)+1] = sentence_t[t]/divisor
        X[i, t+leftpad, len(chars)+2] = sentence_t2[t]/divisor2
        X[i, t+leftpad, len(chars)+3] = sentence_t3[t]/86400
        X[i, t+leftpad, len(chars)+4] = sentence_t4[t]/7
    for c in target_chars:
        if c==next_chars[i]:
            y_a[i, target_char_indices[c]] = 1-softness
        else:
            y_a[i, target_char_indices[c]] = softness/(len(target_chars)-1)
    y_t[i] = next_t/divisor
    np.set_printoptions(threshold=np.nan)

# build the model: 
print('Build model...')
main_input = Input(shape=(maxlen, num_features), name='main_input')
# train a 2-layer LSTM with one shared layer

l1 = LSTM(n_neurons, consume_less='gpu', init='glorot_uniform', return_sequences=True, dropout_W=0.2)(main_input) # the shared layer
b1 = BatchNormalization()(l1)
l_hidden_bn_1=[]
l_hidden_bn_2=[]
l_hidden_1=[]
l_hidden_2=[]
l_hidden_1.append(l1) # the layer specialized in activity prediction
l_hidden_bn_1.append(b1)
l_hidden_2.append(l1) # the layer specialized in time prediction
l_hidden_bn_2.append(b1)
for layer in range(1,n_layers-1):
    l_hidden_1.append(LSTM(n_neurons, consume_less='gpu', init='glorot_uniform', return_sequences=True, dropout_W=0.2)(l_hidden_bn_1[layer-1])) # the layer specialized in activity prediction
    l_hidden_bn_1.append(BatchNormalization()(l_hidden_1[layer]))
    l_hidden_2.append(LSTM(n_neurons, consume_less='gpu', init='glorot_uniform', return_sequences=True, dropout_W=0.2)(l_hidden_bn_2[layer-1])) # the layer specialized in time prediction
    l_hidden_bn_2.append(BatchNormalization()(l_hidden_2[layer]))

l_hidden_1.append(LSTM(n_neurons, consume_less='gpu', init='glorot_uniform', return_sequences=False, dropout_W=0.2)(l_hidden_bn_1[-1])) # the layer specialized in activity prediction
l_hidden_bn_1.append(BatchNormalization()(l_hidden_1[-1]))
l_hidden_2.append(LSTM(n_neurons, consume_less='gpu', init='glorot_uniform', return_sequences=False, dropout_W=0.2)(l_hidden_bn_2[-1])) # the layer specialized in time prediction
l_hidden_bn_2.append(BatchNormalization()(l_hidden_2[-1]))
#l1 = LSTM(100, init='glorot_uniform', return_sequences=True, dropout_W=0.2)(main_input) # the shared layer
#b1 = BatchNormalization()(l1)
#l2_1 = LSTM(100, init='glorot_uniform', return_sequences=False, dropout_W=0.2)(b1) # the layer specialized in activity prediction
#b2_1 = BatchNormalization()(l2_1)
#l2_2 = LSTM(100, init='glorot_uniform', return_sequences=False, dropout_W=0.2)(b1) # the layer specialized in time prediction

act_output = Dense(len(target_chars), activation='softmax', init='glorot_uniform', name='act_output')(l_hidden_bn_1[-1])
time_output = Dense(1, init='glorot_uniform', name='time_output')(l_hidden_bn_2[-1])

model = Model(input=[main_input], output=[act_output, time_output])
model.summary()
# serialize model to JSON
model_json = model.to_json()
with open("output_files/models/"+eventlog+"_"+str(n_neurons)+"_"+str(n_layers)+"_model.json", "w") as json_file:
    json_file.write(model_json)
opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
#opt= Adagrad(lr=0.01, epsilon=1e-08, decay=0.0, clipvalue=3)
model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)
early_stopping = EarlyStopping(monitor='val_loss', patience=42)
model_checkpoint = ModelCheckpoint('output_files/models/'+eventlog+"_"+str(n_neurons)+"_"+str(n_layers)+'_weights_best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
#model_checkpoint = ModelCheckpoint('output_files/models/200_model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

model.fit(X, {'act_output':y_a, 'time_output':y_t}, validation_split=0.2, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=maxlen, nb_epoch=500)
