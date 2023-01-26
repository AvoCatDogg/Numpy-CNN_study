'''
Description: backpropagation operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
'''

import numpy as np #numpy를 import 해줌

from CNN.utils import *

#####################################################
############### Backward Operations #################
#####################################################
        
def convolutionBackward(dconv_prev, conv_in, filt, s):
    
    '''
    Backpropagation through a convolutional layer. 
    convolutional layer 의 역전파과정이다.
    '''

    (n_f, n_c, f, _) = filt.shape #필터는 n_f개와 n_c의 차원, f x f 의 행렬의 구조를 가진다.[]
    (_, orig_dim, _) = conv_in.shape #들어오는 값은 ori_dim x ori_dim의 차원을 가짐.
    #forward과정과 동일하게 필터와 입력값들의 차원을 다음과 같이 받는것이다.

    
    ## initialize derivatives
    dout = np.zeros(conv_in.shape) #dout은 0으로 가득 찬 conv_in의 구조를 가짐. 아래는 다 같은 원리.
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((n_f,1))

    for curr_f in range(n_f):
        # loop through all filters
        # 모든 필터에 대해 반복함.

        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # loss gradient of filter (used to update the filter)
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f]
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                dout[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f] 
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        # loss gradient of the bias
        dbias[curr_f] = np.sum(dconv_prev[curr_f])
    
    return dout, dfilt, dbias



def maxpoolBackward(dpool, orig, f, s):
    '''
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
    maxpooling layer의 역전파 과정이다. 
    '''
    (n_c, orig_dim, _) = orig.shape
    
    dout = np.zeros(orig.shape)
    
    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # obtain index of largest value in input for current window
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y+f, curr_x:curr_x+f])
                dout[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]
                
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return dout
