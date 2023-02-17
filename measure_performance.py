from CNN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='Predict the network accuracy.')
parser.add_argument('parameters', metavar = 'parameters', help='name of file parameters were saved in. These parameters will be used to measure the accuracy.')

if __name__ == '__main__':
    args = parser.parse_args()
    save_path = args.parameters
    
    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    # Get test data
    m =10000
    X = extract_data('./Numpy-CNN_study/cifar-10-batches-py/test_batch', m, 32, 3)
    y_dash = extract_labels('./Numpy-CNN_study/cifar-10-batches-py/test_batch').reshape(m,1)
    # Normalize the data
    for i in range(3): #R G B 따로 Normalize해줌.
        X[:,i]-= int(np.mean(X[:,i])) # subtract mean
        X[:,i]/= int(np.std(X[:,i])) # divide by standard deviation
    
    X=X.reshape(m, 3 * 32 * 32)
    test_data = np.hstack((X,y_dash))
    
    np.random.shuffle(test_data)

    
    X = test_data[0:50,0:-1]
    X = X.reshape(len(test_data[0:50]), 3, 32, 32)
    y = test_data[0:50,-1]

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
        
    print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
    x = np.arange(10)
    digit_recall = [x/(y+0.00001) for x,y in zip(digit_correct, digit_count)]
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(x,digit_recall)
    plt.show()