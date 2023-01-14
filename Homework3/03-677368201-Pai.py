import numpy as np
import matplotlib.pyplot as plt
import struct

# Code to read input
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

train_X = read_idx('train-images-idx3-ubyte')
train_Y = read_idx('train-labels-idx1-ubyte')
test_X = read_idx('t10k-images-idx3-ubyte')
test_Y = read_idx('t10k-labels-idx1-ubyte')

# Helper Functions
def step_func(x1):
    for j in range(0,len(x1)):
        if x1[j] >= 0:
            x1[j] = 1
        else:
            x1[j] = 0
    return x1

def d_x(label):
    temp = np.zeros((10,1))
    temp[label]=1
    return temp

def get_label(v):
    return np.argmax(v)

# Perceptron Training Algorithm
def train_PTA(n,lr,e,W_init):
    epochs = 0
    errors = []
    flag = True
    while flag:
        error = 0
        for i in range(0,n):
            X = train_X[i].reshape(784,1)
            v = np.dot(W_init,X) # Induced Local Field
            label = get_label(v)
            if label != train_Y[i]:
                error+=1
        errors.append(error)
        epochs+=1
        for k in range(0,n): # Weight Update Stage
            X1 = train_X[k].reshape(784,1)
            v = np.dot(W_init,X1)
            W_init = W_init + lr * (d_x(train_Y[k]) - step_func(v)) * X1.T
        # print("Epoch No:",epochs-1,"Error:",error)
        if (errors[epochs-1]/n <= e) or (epochs>=100):
            flag = False
            break
    epochs_1 = [m for m in range(0,len(errors))]
    t = "Perceptron Training for n = "+str(n)+"; learning rate= "+str(lr)+"; e = "+str(e)
    plt.title(t)
    plt.xlabel("Epochs")
    plt.ylabel("Misclassfications")
    plt.plot(epochs_1, errors, color ="green")
    plt.show()
    return W_init

def test_PTA(W_calc,n):
    t_error = 0
    for i in range(0,len(test_X)):
        t_X = test_X[i].reshape(784,1)
        v = np.dot(W_calc,t_X) # Induced Local Field
        label = get_label(v)
        if label != test_Y[i]:
            t_error+=1
    print("Misclassification for Test Set when n is",n,"=",t_error)
    print("Percentage of Misclassification =",(t_error/10000*100))

np.random.seed(2702)
W = np.random.uniform(-1,1,(10,784))

test_PTA(train_PTA(50,1.0,0,W),50)
test_PTA(train_PTA(1000,1.0,0,W),1000)
test_PTA(train_PTA(60000,1.0,0,W),60000)

# Different E
test_PTA(train_PTA(60000,1,0.1375,W),60000)


np.random.seed(100)
W_new = np.random.uniform(-10,10,(10,784))
# 1st time
test_PTA(train_PTA(60000,1,0.11,W_new),60000)
# 2n time
test_PTA(train_PTA(60000,2,0.12,W_new),60000)
# 3rd time
test_PTA(train_PTA(60000,10,0.1,W_new),60000)

        
