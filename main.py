# Homework 1 for PKU DL class
# Daochang Liu, 20180402
# My work includs the following:
# 1. Convolutional autoencoder for feature extraction
# 2. Unsupervised learning using T-SNE and hierarchical clustering
#
# Autoencoder trained on training set of MNIST (60K images)
# Clustering on features of testing set (10K images)
# It takes about 10 mins to run
# Only clustering at the last epoch can save time
# Final accuracy: ~93%

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from itertools import permutations
from model import CNNAutoencoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
import numpy as np
from logger import Logger
from tsne import bh_sne
import matplotlib.pyplot as plt

import pdb

# Hyper Parameters
num_epochs = 8
batch_size = 100
learning_rate = 0.0005
weight_decay = 0.001

logger = Logger('.')

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)  # 60000 28x28 images

test_dataset = dsets.MNIST(root='./data/',
                           train=False, 
                           transform=transforms.ToTensor(),
                           download=True)  # 10000 28x28 images

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
# Data Loader (Input Pipeline)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


encoder = CNNAutoencoder()
encoder.cuda()
encoder.train()
print(encoder)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(encoder.parameters(), 
                             lr=learning_rate,
                             weight_decay=weight_decay)

def train():
    # Train the Model
    encoder.train()

    step = 1
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(train_loader):
            images = Variable(images).cuda()
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs, _ = encoder(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Iter [{}/{}] Loss: {}'.format(
                            epoch+1, num_epochs, i+1, 
                            len(train_dataset)//batch_size, loss.data[0]))

            logger.scalar_summary('loss', loss.data[0], step)
            step += 1

        X, Y = extract_image_codes()
        
        print('Running T-SNE...')
        X_2d = bh_sne(X)
        
        print('Running Hierarchical Clustering...')
        cluster_model = clustering(X_2d)
        plot_scatter(X_2d, cluster_model.labels_, Y, 'Epoch_{}'.format(epoch))

        print('Checking All Label Permutations...')
        acc = get_accuracy(cluster_model.labels_, Y)
        print('Accuracy: {}'.format(acc))
        logger.scalar_summary('Accuracy', acc, epoch+1)


def extract_image_codes():
    encoder.eval()
    idx = 0
    X = np.zeros((10000, 32))
    Y = np.zeros((10000,))
    for images, labels in test_loader:
        images = Variable(images).cuda()
        _, img_code = encoder(images)

        X[idx:idx+batch_size] = img_code.data.cpu().numpy()
        Y[idx:idx+batch_size] = labels.numpy()
        idx += batch_size
    encoder.train()
    return X, Y

def clustering(X):
    #return KMeans(n_clusters=10).fit(X)
    return AgglomerativeClustering(n_clusters=10).fit(X)


def plot_scatter(X, result, Y, name):
    fig = plt.figure(figsize=(20,10))

    plt.subplot(121)
    plt.gca().set_title('Ground Truth')
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=2)

    plt.subplot(122)
    plt.gca().set_title('Clustering Result')
    plt.scatter(X[:, 0], X[:, 1], c=result, s=2)

    fig.savefig(name)
    plt.close(fig)


def get_accuracy(result, Y):

    # hs = metrics.homogeneity_score(Y, result)
    # cs = metrics.completeness_score(Y, result)
    # vms = metrics.v_measure_score(Y, result)
    # print('HS: {}'.format(hs))
    # print('CS: {}'.format(cs))
    # print('VMS: {}'.format(vms))

    confusion_matrix = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            confusion_matrix[i,j] = ((Y==i)*(result==j)).sum()

    best_acc = 0
    perms = list(permutations([i for i in range(10)]))
    for perm in perms:
        accuracy = 0
        for i in range(10):
            accuracy += confusion_matrix[i][perm[i]]
        accuracy = (accuracy / 10000) * 100

        if accuracy > best_acc:
            best_acc = accuracy

    return best_acc


train()
