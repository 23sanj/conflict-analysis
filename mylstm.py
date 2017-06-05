
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import math
import matplotlib.pyplot as plt
from numpy import zeros, newaxis
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.recurrent import LSTM
from matplotlib.backends.backend_pdf import PdfPages

def create_object_func(k):#K= 0-datalist,1-Mlist,2-Rlist,3-nbackslist,4-sublist
    #Reads a file and creates a python list
    data_list=[];#List of all the data_lists
    path = "/Users/sanjana/Documents/MATLAB/Objects-written-in-txt-files/";
    files=os.listdir(path);#listing all the contents in the dir
    with open(path+k) as f:
        content = f.readlines();
        content = [x.strip('\n') for x in content];# you may also want to remove whitespace characters like `\n` at the end of each line
        c=0;
        while c<len(content):
            word = re.split(':|,',content[c]);
            if k == 'subList_obj_names.txt':
                subject_name = re.split('-',word[1]);
                subject_name = subject_name[0];
                data_list.append(subject_name);
                c=c+2;
                continue;
            del(word[len(word)-1]); #stripping the first and the last rows
            del(word[0]);
            for i in range(0,len(word)):
                word[i]=float(word[i]); #Converting strings to float
            data_list.append(word);
           # if c > 0:
            #    data_arr = np.stack((data_arr, word));
            #data_arr = np.asarray(word);
            c=c+2;#Skipping the blank lines
    f.close();
    #fd = open(path+files[k], 'w') 
    #pickle.dump(data_list, fd);
    #Saving the object 
    return data_list;


#function to load the data into numpy arrays
def load_data(sequence_length):
    iData = create_object_func('n_backs_list_obj.txt');#Data input 
    oData = create_object_func('data_list_obj.txt');

    len_seq = []; #List storing lengths of the sequences
    #making sure that all sequences have the same length--this is for the second dimension of the numpy array
    for k in range(len(iData)):#Collecting all input training sequences
        len_seq.append(len(iData[k]));
        for index in range(sequence_length- len(iData[k])):
            iData[k].append(0);#appending 0 so that all have equal length = 220
            oData[k].append(0);

    iData = np.array(iData);
    oData = np.array(oData);
    #the resulting array is 3D with first dimension as the number of sequences, the second as the length of each training seq
    #which is fixed to be 221
    #and the 3rd dimension as the input features(1) + output sequences(1) = 2
    result = np.zeros((len(iData), sequence_length,2));
    result[:,:,0]= iData;
    result[:,:,1]= oData;  

    #Seperating data into training and test sequences
    row = round(1.0*result.shape[0]);
    train = result[:int(row),:];
    #np.random.shuffle(train);#shuffling the sequence along the first axis only

    x_train = train[:,:,0];
    x_train = x_train[:, :, newaxis];
    y_train = train[:,:,1];
    y_train = y_train[:, :, newaxis];

    x_test = result[int(row):,:,0];
    x_test = x_test[:, :, newaxis];
    y_test = result[int(row):,:,1];
    y_test = y_test[:, :, newaxis];

    
    return [x_train, y_train, x_test, y_test,len_seq];

#function to build the network:
def build_network(batch_size,epoch,hidden_neurons):
    
    model = Sequential(); #To stack layers one on top of another
    #layers =[1,50,100,1];#List containing the size of each layer--1D input and 1D output and hidden layer of sizes 50 and 100 neurons
    #hidden_neurons =50;
    #LSTM 1st layer:
    model.add(LSTM(hidden_neurons,input_shape=(None, 1),return_sequences=True));#Creating the first LSTM layer
    model.add(Dropout(0.2));

    #LSTM 2nd layer
    #model.add(LSTM(layers[2],return_sequences=False));
    #model.add(Dropout(0.2));

    #Last layer is a Dense= feedforward layer
    model.add(TimeDistributed(Dense(1)));# each LSTM time step to predict a single scalar --accuracy output
    model.add(Activation("sigmoid"));#softmax on a layer with 1 node will always output 1.0

    model.compile(loss='mse', optimizer='rmsprop');
    #pr = model.predict(x_train);
    #predicted = np.reshape(predicted,(predicted.size,));
    
    return model;

def plot_trace(len_seq):
    n_backs = create_object_func('n_backs_list_obj.txt');
    subject_names = create_object_func('subList_obj_names.txt');
    pp = PdfPages('User_Skill_trace_RNN.pdf');
    nSubs =len(len_seq);
    for i in range(nSubs):
    	lseq = len_seq[i];
    	actual_y = y_train[i,:lseq,0];
    	pred = pr[i,:lseq,0];
    	sessions = np.linspace(1,lseq,lseq);
    	fig = plt.figure(facecolor ='white');
    
    	ax = fig.add_subplot(111);
    
    	ax.scatter(sessions,actual_y,c='r', marker='x',s=10,label='Actual Classification Ratio');
    	ax.scatter(sessions,pred,facecolors='none', edgecolors='b', marker='o',s=5,label='Predicted Classification Ratio');
    
    	for k,txt in enumerate(n_backs[i]):
        	str(txt);
        	ax.annotate(txt, (sessions[k],actual_y[k]),size=5);
        	#ax.annotate(txt, (sessions[k],pred[k]));
    
    	#ax.plot(y_test,label='Actual Classification Ratio');
    	#ax.plot(pr,label='Predicted Classification Ratio');
    	plt.legend();
    	plt.title(subject_names[i]);
    	plt.show();
    	pp.savefig(fig);

    pp.close();
    return;

def compute_rmse(y_train,pr,len_seq,sequence_length):
    nSubs = len(len_seq);
    residuals = np.zeros((len(len_seq), sequence_length));

    for i in range(len(len_seq)):
        lseq = len_seq[i];
        actual_y = y_train[i,:,0];
        pred = pr[i,:,0];
        sessions = np.linspace(1,lseq,lseq);

        residuals[i,:] = abs(actual_y - pred); #Collecting the errors in a 2d array

    residuals_sqr = np.square(residuals);

    #rmse = sqrt(sum(residuals.*residuals)/length(residuals));
    rmse_sum =residuals_sqr.sum(axis=0, dtype='float');#umming the residuals across all subjects for a time-step
    rmse_res = sum(rmse_sum)/(sum(len_seq));
    rmse_res = rmse_res**(1.0/2)
    fig=plt.figure();
    plt.bar(np.linspace(1,sequence_length,sequence_length), (rmse_sum/nSubs), color="blue")
    #plt.plot(rmse_sum/nSubs);
    plt.show();
    fig.savefig('rmse.png');

    #Plotting the mean and sd of the errors:
    sum_subj =residuals.sum(axis=0, dtype='float');
    mean_res = sum_subj/nSubs;   
    fig=plt.figure();
    plt.bar(np.linspace(1,sequence_length,sequence_length),mean_res, color="blue")
    plt.show();
    fig.savefig('mean.png');

    var_res = np.zeros(sequence_length);
    for i in range(nSubs):
        res1 =(residuals[i,:] -mean_res);
        var_res = (res1*res1) + var_res;

    var_res = var_res/nSubs;
    sd_res = (var_res)**(1.0/2)

    fig=plt.figure();
    plt.bar(np.linspace(1,sequence_length,sequence_length),sd_res, color="blue")
    plt.show();
    fig.savefig('sd.png');
    return rmse_res;

if __name__ == "__main__":
    print('Input in order:\n the length of sequence \n number of epochs for training\n number of hidden units in Layer 1\n ');
   
    batch_size = int(sys.argv[1]);
    epoch = int(sys.argv[2]);
    hidden_neurons = int(sys.argv[3]);

    x_train, y_train, x_test, y_test,len_seq = load_data(batch_size);
    print('Data Loaded--- Compiling');

    model = build_network(batch_size,epoch,hidden_neurons);
    model.fit(x_train,y_train,batch_size,epoch);
    
    pr = model.predict(x_train); #Predicting on the whole training dataset
    
    #Computing errors:
    rmse=compute_rmse(y_train,pr,len_seq,batch_size);
    #Plotting the trace:
    plot_trace(len_seq);
    



   
    