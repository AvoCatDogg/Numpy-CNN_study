'''
Description: Utility methods for a Convolutional Neural Network

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''

from CNN.forward import * #앞서 코드를 작성한  forward를 import함
import numpy as np
import gzip  #파일 압축할때 쓰는 응용소프트웨어
import matplotlib.pyplot as plt
import pickle

#####################################################
################## Utility Methods ##################
#####################################################
        
def extract_data(filename, num_images, IMAGE_WIDTH, img_depth):
    '''
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w], where m 
    is the number of training examples.
    이미지를 파일의 Bytestream을 읽음으로서 추출한다. 읽어준 값을 [m,h,w]의 3차원 행렬로 만들어준다. 
    m은 traning examples의 수를 의미한다.  
    '''

    print('Extracting', filename,'data')
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    data = np.array(dict[b'data']).reshape(num_images, img_depth , IMAGE_WIDTH * IMAGE_WIDTH).astype(np.float32) #이미지 갯수, 색, 32 X 32

    return data

    '''
    #원래 코드
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data
    '''

    #이쪽 위에부분들이 이해가 안됨. gzip.read(숫자) 에서 숫자의 의미를 모르겠다.
    #np.frombuffer은 binary형태의 데이터를 array로 받는것

def extract_labels(filename):

    '''
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    [m,1] 차원의 행렬로 label을 추출한다. m은 image의 수이다.
    '''
    print('Extracting', filename,'data')
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    labels = np.array(dict[b'labels']).astype(np.int64)
    
    return labels

    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels
    '''

def initializeFilter(size, scale = 1.0): #필터를 초기화 하는 코드
    stddev = scale/np.sqrt(np.prod(size)) # size라는 array의 요소들을 곱한 값의 sqrt를 크기로 나누어준다.
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size): #가중치를 초기화 한다.
    return np.random.standard_normal(size=size) * 0.01

def nanargmax(arr): 
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs    

def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s = 1, pool_f = 2, pool_s = 2):
    '''
    Make predictions with trained filters/weights. 
    '''
    conv1 = convolution(image, f1, b1, conv_s) # convolution operation
    conv1[conv1<=0] = 0 #relu activation
    
    conv2 = convolution(conv1, f2, b2, conv_s) # second convolution operation
    conv2[conv2<=0] = 0 # pass through ReLU non-linearity
    
    pooled = maxpool(conv2, pool_f, pool_s) # maxpooling operation
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
    
    z = w3.dot(fc) + b3 # first dense layer
    z[z<=0] = 0 # pass through ReLU non-linearity
    
    out = w4.dot(z) + b4 # second dense layer
    probs = softmax(out) # predict class probabilities with the softmax activation function
    
    return np.argmax(probs), np.max(probs)
    