import numpy as np
from CNN.utils import *
import matplotlib.pyplot as plt
import pickle
m=10000
img_dim=32
img_depth=3

with open("Numpy-CNN_study\cifar-10-batches-py\data_batch_1", 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

data = np.array(dict[b'data'])
image_set = extract_data("Numpy-CNN_study\cifar-10-batches-py\data_batch_1",m,img_dim,img_depth).reshape(m,32*32*3)
label_set = extract_labels("Numpy-CNN_study\cifar-10-batches-py\data_batch_1")

label_name =np.array(['airplane','auto','bird','cat','deer','dog','deer','dog','frog','horse'])
label_index=np.zeros(10)

for i in range(10):
    A=np.where(label_set==i)
    np.random.shuffle(A[0])
    label_index[i]=A[0][0]
label_index=np.array(label_index).astype(np.int64)

image_set=image_set.reshape(m,img_depth,img_dim*img_dim)
image_set_r = image_set[:,0,:]
image_set_g = image_set[:,1,:]
image_set_b = image_set[:,2,:]

image=np.dstack((image_set_r,image_set_g,image_set_b))
image=image.reshape(m,img_dim,img_dim,img_depth).astype(dtype = np.int64)


for i in range(10):
    img=image[label_index[i]]
    print(label_name[i])
    plt.imshow(img,interpolation='none', aspect='auto')
    
