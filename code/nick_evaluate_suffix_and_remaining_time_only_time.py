'''
this script takes as input the LSTM or RNN weights found by train.py
change the path in line 178 of this script to point to the h5 file
with LSTM or RNN weights generated by train.py

Author: Niek Tax
'''

from __future__ import division
from keras.models import load_model
import csv
import copy
import numpy as np
import distance
from itertools import izip
from jellyfish._jellyfish import damerau_levenshtein_distance
import unicodecsv
from sklearn import metrics
from math import sqrt
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter
from keras.models import model_from_json
import sys
fileprefix=sys.argv[1]
eventlog = sys.argv[2]
csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
ascii_offset = 161

lastcase = ''
line = ''
firstLine = True
lines = []
timeseqs = []
timeseqs2 = []
timeseqs3 = []
timeseqs4 = []
y_times=[]
times = []
times2 = []
times3 = []
times4 = []
#nick
attributes=[]
attributes_dict=[]
attributes_sizes=[]



numlines = 0
casestarttime = None
lasteventtime = None

csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
ascii_offset = 161
y=[]
for row in spamreader:
    #print(row)
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    #test different format
    #t = 0#time.strptime(row[2], "%Y/%m/%d %H:%M:%S")

    if row[0]!=lastcase:
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
            #target
            y_times.extend([times2[-1]-k for k in times2])
            timeseqs3.append(times3)
            timeseqs4.append(times4)
            for i in xrange(len(attributes)):
                attributes[i].append(attributesvalues[i])
        else:
            #if firstline. I have to add te elements to attributes
            for a in row[3:]:
                attributes.append([])
                attributes_dict.append({})
                attributes_sizes.append(0)
        #print(attributes)
        n_events_in_trace=0
        line = ''
        times = []
        times2 = []
        times3 = []
        times4 = []

        numlines+=1
    n_events_in_trace+=1
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
    attributesvalues=[]
    indexnick=0
    for a in row[3:]:
        #todo cast a intero se e intero if
        if a in attributes_dict[indexnick]:
            attributesvalues.append(attributes_dict[indexnick][a])
        else:
            attributes_dict[indexnick][a]=attributes_sizes[indexnick]
            attributes_sizes[indexnick]+=1
            attributesvalues.append(attributes_dict[indexnick][a])

        indexnick+=1

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
timeseqs3.append(times3)
timeseqs4.append(times4)
y_times.extend([times2[-1] - k for k in times2])
for i in xrange(len(attributes)):
    attributes[i].append(attributesvalues[i])
numlines+=1



divisor = np.mean([item for sublist in timeseqs for item in sublist])
print('divisor: {}'.format(divisor))
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))
divisor3 = np.mean(map(lambda x: np.mean(map(lambda y: x[len(x)-1]-y, x)), timeseqs2))
print('divisor3: {}'.format(divisor3))

elems_per_fold = int(round(numlines/3))
fold1 = lines[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]
#nick
fold1_a=[a[:elems_per_fold] for a in attributes]

fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2*elems_per_fold]
#nick
fold2_a=[a[elems_per_fold:2*elems_per_fold] for a in attributes]

fold3 = lines[2*elems_per_fold:]
fold3_t = timeseqs[2*elems_per_fold:]
fold3_t2 = timeseqs2[2*elems_per_fold:]
#nick
fold3_a=[a[2*elems_per_fold:] for a in attributes]

lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2

step = 1
sentences = []
softness = 0
next_chars = []
#lines = map(lambda x: x+'!',lines)
maxlen = max(map(lambda x: len(x),lines))

chars = map(lambda x : set(x),lines)
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars)
#chars.remove('!')
print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
print(indices_char)



fold3 = lines[2*elems_per_fold:]
fold3_t = timeseqs[2*elems_per_fold:]
fold3_t2 = timeseqs2[2*elems_per_fold:]
fold3_t3 = timeseqs3[2*elems_per_fold:]
fold3_a=[a[2*elems_per_fold:] for a in attributes]

lines = fold3
lines_t = fold3_t
lines_t2 = fold3_t2
lines_t3 = fold3_t3

# set parameters
predict_size = maxlen


# load json and create model
json_file = open('output_files/models/'+fileprefix+'_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("output_files/models/"+fileprefix+"_weights_best.h5")
print("Loaded model from disk")

y_t_seq=[]

# load model, set this to the model generated by train.py
#model = load_model('output_files/models/200_model_59-1.50.h5')

# define helper functions
def encode(sentence, times, times3, sentences_attributes, y_t_seq,maxlen=maxlen):
    num_features = len(chars)+5+len(sentences_attributes)
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen-len(sentence)
    times2 = np.cumsum(times)
    print "sentence",len(sentence)
    for t, char in enumerate(sentence):
        midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = times3[t]-midnight
        multiset_abstraction = Counter(sentence[:t+1])
        for c in chars:
            if c==char:
                X[0, t+leftpad, char_indices[c]] = 1
        X[0, t+leftpad, len(chars)] = t+1
        X[0, t+leftpad, len(chars)+1] = times[t]/divisor
        X[0, t+leftpad, len(chars)+2] = times2[t]/divisor2
        X[0, t+leftpad, len(chars)+3] = timesincemidnight.seconds/86400
        X[0, t+leftpad, len(chars)+4] = times3[t].weekday()/7
        for i in xrange(len(sentences_attributes)):
            print(str(i)+" "+str(t))
            print(sentences_attributes[i][t])
            #nick check the zero, it is there because it was a list
            X[0, t + leftpad, len(chars) + 5+i]=sentences_attributes[i][t]
    y_t[i] = y_t_seq[i][-1]/divisor
    return X,y

def getSymbol(predictions):
    maxPrediction = 0
    symbol = ''
    i = 0;
    for prediction in predictions:
        if(prediction>=maxPrediction):
            maxPrediction = prediction
            symbol = target_indices_char[i]
        i += 1
    return symbol

one_ahead_gt = []
one_ahead_pred = []

two_ahead_gt = []
two_ahead_pred = []

three_ahead_gt = []
three_ahead_pred = []

y_t_seq=[]

# make predictions
with open('output_files/results/'+fileprefix+'_suffix_and_remaining_time_%s' % eventlog, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard", "Ground truth times", "Predicted times", "RMSE", "MAE", "Median AE"])
    #considering also size 1 prefixes
    for prefix_size in range(1,maxlen):
        #print(prefix_size)
        for line, times, times2, times3 in izip(lines, lines_t, lines_t2, lines_t3):
            times.append(0)
            cropped_line = ''.join(line[:prefix_size])
            cropped_times = times[:prefix_size]
            print "times_len",len(cropped_times)
            cropped_times3 = times3[:prefix_size]
            cropped_attributes = [[] for i in xrange(len(attributes))]
            for a in xrange(len(attributes)):
                # print(attributes[a][0:i])
                cropped_attributes[a].extend(attributes[a][0:prefix_size])
            y_t_seq.append(y_times[0:i])

            #cropped_attributes= [a[:prefix_size] for a in attributes]
            #print cropped_attributes

            if len(times2)<prefix_size:
                continue # make no prediction for this case, since this case has ended already
            ground_truth = ''.join(line[prefix_size:prefix_size+predict_size])
            ground_truth_t = times2[prefix_size-1]
            case_end_time = times2[len(times2)-1]
            ground_truth_t = case_end_time-ground_truth_t
            predicted = ''
            total_predicted_time = 0

            #perform single prediction
            enc = encode(cropped_line, cropped_times, cropped_times3,cropped_attributes)
            y = model.predict(enc, verbose=0) # make predictions
            # split predictions into seperate activity and time predictions
            print y
            y_t = y[0][0][0]
            prediction = getSymbol(y_char) # undo one-hot encoding
            cropped_line += prediction
            if y_t<0:
                y_t=0
            cropped_times.append(y_t)

            y_t = y_t * divisor3
            cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
            total_predicted_time = total_predicted_time + y_t

            output = []
            if len(ground_truth)>0:
                output.append(prefix_size)
                output.append(unicode(ground_truth).encode("utf-8"))
                output.append(ground_truth_t)
                output.append(total_predicted_time)
                output.append('')
                output.append(metrics.mean_absolute_error([ground_truth_t], [total_predicted_time]))
                output.append(metrics.median_absolute_error([ground_truth_t], [total_predicted_time]))
                spamwriter.writerow(output)
