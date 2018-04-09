
# coding: utf-8

# In[1]:

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    medical_files = np.array(data['filenames'])
    medical_targets = np_utils.to_categorical(np.array(data['target']), 3)
    return medical_files, medical_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('D:\data\medical AI\data/train')
valid_files, valid_targets = load_dataset('D:\data\medical AI\data/valid')
test_files, test_targets = load_dataset('D:\data\medical AI\data/test')


# In[2]:

#数据处理，使用Keras的办法将图片统一转换为768*512*3的尺寸
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# In[3]:

dog_names = [item[30:-1] for item in sorted(glob("D:\data\medical AI\data/train/*/"))]


# In[4]:

dog_names


# In[3]:

#数据处理，将统一后的图片转换为批向量,并进行归一化
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
#train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
#test_tensors = paths_to_tensor(test_files).astype('float32')/255


# In[4]:

import tensorflow as tf
#搭建CNN网络
def model_variables(weight=224, height=224,channels=3):
    input_ = tf.placeholder(tf.float32, (None, weight, height, channels), name='input')
    labels = tf.placeholder(tf.int32, (None, channels), name='labels')
    lr = tf.placeholder(tf.float32)
    return input_, labels, lr

def build_model(input_, alpha=0.2):       
    
        
    x1 = tf.layers.conv2d(input_, filters=16, kernel_size=5, strides=2, padding='same', 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
    x1 = tf.layers.batch_normalization(x1)
    x1 = tf.maximum(0.2*x1, x1)
    x1 = tf.nn.dropout(x1, keep_prob = 0.5)
                #256*256*16

    x2 = tf.layers.conv2d(x1, filters=32, kernel_size=5, strides=2, padding='same',
                                          kernel_initializer = tf.contrib.layers.xavier_initializer())
    x2 = tf.layers.batch_normalization(x2)
    x2 = tf.maximum(0.2* x2, x2)
    x2 = tf.nn.dropout(x2, keep_prob = 0.5)
                #128*128*32

    x3 = tf.layers.conv2d(x2, filters=64, kernel_size=5, strides=2, padding='same',
                                          kernel_initializer = tf.contrib.layers.xavier_initializer())
    x3 = tf.layers.batch_normalization(x3)
    x3 = tf.maximum(0.2*x3, x3)
    x3 = tf.nn.dropout(x3, keep_prob = 0.5)
                #64*64*64

    x4 = tf.reshape(x3,(-1,28*28*64))
    logits = tf.layers.dense(x4, 3)
    output = tf.nn.softmax(logits)
    return logits, output

def loss(logits, labels):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=labels))
    return loss

def opt(loss,learning_rate, beta1=0.8):
    opt = tf.train.AdamOptimizer(learning_rate=0.01, beta1 =0.8).minimize(loss)
    return opt


# In[6]:

def get_batches(filepath, targets, batch_size):
    whole_size = filepath.shape[0]
    num_batches = whole_size//batch_size
    for i in range(num_batches):
        yield paths_to_tensor(filepath[i*batch_size:(i+1)*batch_size]).astype('float32')/255, targets[i*batch_size:(i+1)*batch_size]


# In[56]:

epoches = 10
batch_size = 100
learning_rate = 0.005





# In[57]:

input_, labels, lr = model_variables()
logits, output = build_model(input_)
loss_ =loss(logits, labels)
opt_  = opt(loss_,learning_rate)
steps = 0
saver = tf.train.Saver()
train_loss = []
valid_loss = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(epoches):
        for train_tensors, train_targets_batch in get_batches(train_files,train_targets,batch_size):
            #for batch_img, batch_label in get_batch(train_data_path):
            #_ = sess.run(loss_, )
            _,__ = sess.run([loss_,opt_],feed_dict={input_:train_tensors, labels:train_targets_batch})
            steps += 1
            train_loss.append(_)
            print('{}:train loss is {}'.format(steps,_))
            
                
            if steps%10==0:
                _ = sess.run(loss_, feed_dict={input_:valid_tensors, labels:valid_targets})
                if valid_loss ==0 or valid_loss>= _:
                    print('valid_loss improves')
                    valid_loss = _
                    saver.save(sess, './checkpoints/medical.ckpt')
                print('{}: valid_loss is{}'.format(steps, _))


# In[26]:

result = 0


# In[27]:

#测试模型准确率
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# In[58]:

with tf.Session() as sess:
    saver.restore(sess, './checkpoints/medical.ckpt')
    test_output = sess.run(output, feed_dict={input_:test_tensors})
    print(test_output.shape)
    result = test_output
    test_accuracy = np.sum(np.argmax(test_output,axis=1)==np.argmax(test_targets,axis=1))/600.0
    print(test_accuracy)


# In[52]:

from pandas import DataFrame


# In[53]:

DataFrame(result).to_csv('zuichu.csv')


# In[54]:

DataFrame(test_files).to_csv('zuichu1.csv')


# In[ ]:



