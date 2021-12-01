import numpy as np
import pandas as pd
from pdb import set_trace as st
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from model import CNN
from gen_dataset import my_dataset
import os
import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms




def main():


    ### training network
    if not os.path.exists('../model/'):
        os.mkdir('../model/')


    ### Hyperparameters
    BATCH_SIZE = 128
    EPOCH = 150
    LEARNING_RATE = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Now using device {device}!')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print('Preparing dataset:')
    ### load dataset
    data = pd.read_csv('../data/archive/hmnist_28_28_RGB.csv')

    ### imblearn, labels amount are not even. So do data augumentation before training network
    ros = RandomOverSampler()
    labels = data['label']
    images = data.drop(columns=['label'])
    images, labels = ros.fit_resample(images, labels)
    # print(f'Before balancing {Counter()}')
    # print(f'After balancing {Counter(labels)}')

    print(f'The size of dataset {images.shape[0]}')

    X_train, X_eval, y_train, y_eval = train_test_split(images, labels, train_size=0.8, random_state=21)
    y_train = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)
    y_eval = torch.from_numpy(np.array(y_eval)).type(torch.LongTensor)

    train_data = my_dataset(df=X_train, labels=y_train, transform=transform)
    eval_data = my_dataset(df=X_eval, labels=y_eval, transform=transform)

    data_loader_train = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    data_loader_eval = DataLoader(eval_data, batch_size=BATCH_SIZE, shuffle=True)

    ### training
    using_model = CNN().to(device)
    print('Model summary:')
    summary(using_model, input_size=(3,28,28))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(using_model.parameters(), lr=LEARNING_RATE, betas=(0.9,0.999),eps=1e-8)


    for epoch in range(1,EPOCH+1):
        train_loss = 0
        train_acc = 0
        train_step = 0
        using_model.train()
        for image, label in tqdm.tqdm(data_loader_train):
            image = Variable(image.to(device))
            label = Variable(label.to(device))

            output = using_model(image)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct/BATCH_SIZE
            train_acc += acc
            train_step += 1
        train_loss /= train_step
        train_acc /= train_step

        ## eval
        eval_loss = 0
        eval_acc = 0
        eval_step = 0
        using_model.eval()
        for image, label in data_loader_eval:
            image = Variable(image.to(device))
            label = Variable(label.to(device))

            output = using_model(image)
            loss = criterion(output, label)

            eval_loss += loss.item()
            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / BATCH_SIZE
            eval_acc += acc
            eval_step += 1

        eval_loss /= eval_step
        eval_acc /= eval_step

        print(f'Epoch {epoch}/{EPOCH}: Train Loss: {train_loss}, Train Acc: {train_acc}, Eval Loss: {eval_loss}, Eval Acc: {eval_acc}')

        torch.save(using_model.state_dict(),'../model/cnn_epoch{}.pth'.format(epoch))

    print('Done!')


if __name__ == '__main__':
    main()
