'''
Description: forward operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
'''
import numpy as np

#####################################################
################ Forward Operations #################
#####################################################

def convolution(image, filt, bias, s=1): #convolution함수를 정의해줌

    '''
    Convolves `filt` over `image` using stride `s`
    이미지에 필터를 합성곱하는데, 이때의 stride는 `s`임
    '''

    (n_f, n_c_f, f, _) = filt.shape # filter dimension Q: n_f, n_c_f의 정확한 의미, 그리고 f, _의 의미
                                    # 필터의 차원 (필터갯수, 필터차원,fxf 행렬)

    n_c, in_dim, _ = image.shape # image dimensions 
                                 # 이미지의 차원 (이미지 차원, in_dim x indim 행렬)

    out_dim = int((in_dim - f)/s)+1 # calculate output dimensions 
                                    #output의 차원을 계산함.
                                    #이와같은 차원을 가지는 이유는 계산을 하면 됨. stride에 따른 값임
    
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    #assert는 조건문이 False일때 출력되는것이다. 필터의 차원이 인풋 이미지의 차원과 같아야 한다는 의미이며, 조건문 또한 이 의미임
    
    out = np.zeros((n_f,out_dim,out_dim)) #
    
    #convolve the filter over every part of the image, adding the bias at each step. 
    #이미지의 모든 파트들에 대해 필터를 convolve 할거임. 근데, 모든 단계에서 bias를 더해줄거다.
    
    for curr_f in range(n_f): #curr_f 를 n_f까지 반복

        curr_y = out_y = 0    #curr_f 와 out_y 를 0으로 리셋
        while curr_y + f <= in_dim: #이미지 y 길이보다, 필터길이+현재 필터위치가 작은 경우에 실행
            curr_x = out_x = 0      #현재 필터의 x위치를 0으로 초기화

            while curr_x + f <= in_dim:#이미지 x길이보다 필터+현재 필터 위치가 작은 경우에 실행
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f] 
                #위 과정이 합성곱의 과정이다. current f값에서의 filt값을 f크기의 격자에 들어있는 image에 모든 항 : 과 더하고, 이에 bias를 더함
                curr_x += s #현재 필터 위치에 stride를 더해줌
                out_x += 1 #다음 결과값을 받을 out행렬의 x위치를 정해줌.
            curr_y += s #위와 같음
            out_y += 1
        
    return out #convolution의 결과를 out으로 받음.


def maxpool(image, f=2, s=2): #maxpool은 앞서 필터를 거친 값에서 karnel안의 가장 큰 값을 뽑아내는 과정이라 생각하면 된다.

    '''
    Downsample `image` using kernel size `f` and stride `s`
    '''

    n_c, h_prev, w_prev = image.shape #이미지 차원, height, width
    
    h = int((h_prev - f)/s)+1 #차원은 filter에 의해서 다음과 같이 됨.
    w = int((w_prev - f)/s)+1
    downsampled = np.zeros((n_c, h, w)) #처리된 샘플을 미리 zeros로 정의해줌

    for i in range(n_c):
        # slide maxpool window over each part of the image and assign the max value at each step to the output
        # maxpool 칸막이를 이미지의 각각의 칸으로 민다, 그리고 모든 값들의 최고값을 각각의 단계에 부여함. 
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:

                #앞선 합성곱 과정과 같은 원리로 작동함.
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                #downsampled는 image에 f by f 의 칸막이중 가장 큰 값을 받게됨.

                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled

def softmax(X): #소프트맥스 함수의 정의
    out = np.exp(X) 
    return out/np.sum(out)

def categoricalCrossEntropy(probs, label): #CCE를 정의
    return -np.sum(label * np.log(probs))

