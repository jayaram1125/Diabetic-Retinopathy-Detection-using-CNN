import tensorflow as tf
import zipfile as zf
import numpy as np
from PIL import Image
import os
import pandas as pd
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 

i = 1
data = []
labels = []




df = pd.read_csv('trainLabels.csv')

#Read the input data from the Images folder
for root,dirs,files in os.walk('Images'):
   for file in files:
      if file.endswith('.jpeg'):
         s = file.split('.')[0]
         dfrow = df[df['image']==s].index.values.astype(int)[0]
         labels.append(df.iloc[dfrow]['level'])
         imgpath = os.path.join(root,file)
         img = Image.open(imgpath)
         print(imgpath)
         data.append(np.array(img).reshape(512,512,3))
         print(len(data))
         i = i+1

print("datarralen="+str(len(data)))
print("labelarrlen="+str(len(labels)))




labelsarr = np.array(labels).reshape(len(labels),1)
dataarr = np.array(data)

print(dataarr.shape)
print(labelsarr.shape)

totallen = len(dataarr)

X_train = dataarr[0:int(0.8*totallen)]
X_test =  dataarr[int(0.8*totallen):totallen]

Y_train_lab = labelsarr[0:int(0.8*totallen)] 
Y_test_lab = labelsarr[int(0.8*totallen):totallen]



def OneHotEncoding(labeldata,classsize):
   retarr = np.zeros((len(labeldata),classsize))
   for i in range(0,len(labeldata)):
     retarr[i][labeldata[i]]=1

   return retarr

Y_train = OneHotEncoding(Y_train_lab,5)
Y_test = OneHotEncoding(Y_test_lab,5)



print(X_train.shape)
print(X_test.shape)

print(Y_train.shape)
print(Y_test.shape)



x = tf.placeholder("float", shape=[None,512,512,3])
y_ = tf.placeholder("float", shape=[None,5])
keep_prob = tf.placeholder("float")


def conv2d(input_layer,ifilters,ikernel_size,istrides):
    layer = tf.layers.conv2d(inputs=input_layer,
      				      filters=ifilters,
				      kernel_size= ikernel_size,
                                      strides = istrides,
      				      padding="same",
				      activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(1.0),
                                      bias_initializer = tf.constant_initializer(0.1))
    return layer


def maxpool2d(input_layer,ipool_size,istrides):
    layer = tf.layers.max_pooling2d(inputs = input_layer,
				    pool_size = ipool_size,
				    strides = istrides,
				    padding= 'same')
    return layer   


def build_model():
    x_image = tf.reshape(x,[-1,512,512,3])
       
    layer = conv2d(x_image,32,[7,7],[2,2])
    layer =  maxpool2d(layer,[3,3],[2,2])  
          
    layer = conv2d(layer,32,[3,3],[1,1])
    layer = conv2d(layer,32,[3,3],[1,1]) 
    layer =  maxpool2d(layer,[3,3],[2,2])   
       
   
    layer = conv2d(layer,64,[3,3],[1,1])
    layer = conv2d(layer,64,[3,3],[1,1])  
    layer =  maxpool2d(layer,[3,3],[2,2])  

    layer = conv2d(layer,128,[3,3],[1,1])
    layer = conv2d(layer,128,[3,3],[1,1]) 
    layer = conv2d(layer,128,[3,3],[1,1])
    layer = conv2d(layer,128,[3,3],[1,1]) 
    layer =  maxpool2d(layer,[3,3],[2,2])  

    layer = conv2d(layer,256,[3,3],[1,1])
    layer = conv2d(layer,256,[3,3],[1,1]) 
    layer = conv2d(layer,256,[3,3],[1,1])
    layer = conv2d(layer,256,[3,3],[1,1]) 
    layer =  maxpool2d(layer,[3,3],[2,2])  
  
    layer = tf.layers.dropout(layer,0.5)        
 
    layer = tf.layers.Flatten()(layer)  
     
    layer = tf.layers.dense(layer,
                        1024,
   			kernel_initializer = tf.orthogonal_initializer(1.0),
   		        bias_initializer = tf.constant_initializer(0.1))              

    layer = tf.nn.relu(layer)
    layer = tf.layers.dropout(layer,0.5) 

    layer = tf.layers.dense(layer,
                        5,
   			kernel_initializer = tf.orthogonal_initializer(1.0),
   		        bias_initializer = tf.constant_initializer(0.1)) 
    layer =  tf.nn.softmax(layer)

    return layer


def next_batch(batch_size, data, labels,offset):
    batchX = data[offset:(offset+batch_size),:]
    batchY = labels[offset:(offset+batch_size),:] 
    return batchX,batchY

def train():    
    batch_size = 20
    for i in range(0,2000): 
        offset =(i*batch_size)%(len(X_train)-batch_size)  
        batch = next_batch(batch_size,X_train,Y_train,offset)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={ x: batch[0], y_: batch[1], keep_prob: 1.0})  
            print "Step: %d, Training accuracy = %g" % (i, train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
 
def testandPrintAccuracy():
    batch_size = 20
    for i in range(0,1000):
       offset =(i*batch_size)%(len(X_test)-batch_size)
       batch = next_batch(batch_size,X_test,Y_test,offset)
       if i%100 == 0:
           print "Step %d, Test accuracy = %g" %(i, accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})) 

model = build_model()
cross_entropy = -tf.reduce_sum(y_ * tf.log(model))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()
sess = tf.InteractiveSession(config=config)
sess.run(init)

train()
testandPrintAccuracy()
