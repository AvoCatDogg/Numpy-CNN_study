'''
Description: Script to train the network and measure its performance on the test set.

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''

from CNN.network import *
from CNN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='Train a convolutional neural network.')
parser.add_argument('save_path', metavar = 'Save Path', help='name of file to save parameters in.')

#학습을 시키는 main이다.

if __name__ == '__main__':
    args = parser.parse_args()
    save_path = args.save_path
    
    cost = train(save_path = save_path)

    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    # Plot cost
    plt.plot(cost, 'r')
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend('Loss', loc='upper right')

    plt.show()

    # Get test data
    m =10000
    IMAGE_WIDTH = 32
    img_depth = 3
    data_size = 5 # datasize 는 이 값의 * 10000이라 생각하자. 즉, 열어보는 data_batch_n 파일의 갯수임. max =5

    test_data = select_train_data(data_size = 5, random = False, Scaling_Style = 'standard')
    '''
    X = extract_data('./Numpy-CNN_study/cifar-10-batches-py/data_batch_{i}', num_image = 10000, IMAGE_WIDTH = 32, img_depth = 3)
    y_dash = extract_labels('./Numpy-CNN_study/cifar-10-batches-py/data_batch_{i}').reshape(m,1)
    
    # Normalize the data
    for i in range(3): #R G B 따로 Normalize해줌.
        X[:,i]-= int(np.mean(X[:,i])) # subtract mean
        X[:,i]/= int(np.std(X[:,i])) # divide by standard deviation
    
    X=X.reshape(m, img_depth * IMAGE_WIDTH * IMAGE_WIDTH)
    y=y_dash

    
    test_data = np.hstack((X,y_dash))
    '''
    
    X = test_data[:,0:-1]
    X = X.reshape(len(test_data), 3, 32, 32)
    y = test_data[:,-1]
    
    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]
   
    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
            digit_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
        
    print("Overall Accuracy: %.2f" % (float(corr/len(y)*100)))
    x = np.arange(10)
    digit_recall = [x/y for x,y in zip(digit_correct, digit_count)]
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(x,digit_recall)
    plt.show()
