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
import time
import psutil
from glob import glob
import sys


def main():

    if(len(sys.argv) < 4):
        print(f"Usage: python3 {sys.argv[0]} model_folder data_folder result_csv_saved_path")
        print(f"Example: python3  {sys.argv[0]} ./saved_model ../data/ ./evaluation_result.csv")
        exit()


    ### Hyperparameters
    BATCH_SIZE = 128
    EPOCH = 100
    LEARNING_RATE = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Now using device {device}!')




    model_saved_path = sys.argv[1]
    model_list = glob(os.path.join(model_saved_path, '*.pth'))
    model_list = np.sort(model_list)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


    using_model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    result_csv_saved_path = sys.argv[3]

    data_folder = sys.argv[2]
    for data_path in [os.path.join(data_folder, 'hmnist_28_28_RGB_train.csv'), os.path.join(data_folder, 'hmnist_28_28_RGB_test.csv')]:
    # data_list = glob(os.path.join(data_folder, 'hmnist_28_28_RGB_*'))

        data = pd.read_csv(data_path)

        print('Preparing dataset:')
        ### load dataset

        ### imblearn, labels amount are not even. So do data augumentation before training network
        # ros = RandomOverSampler()
        labels = data['label']
        images = data.drop(columns=['label'])
        # images, labels = ros.fit_resample(images, labels)

        print(f'The size of dataset {images.shape[0]}')



        X_eval = images
        y_eval = labels

        # y_train = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)
        y_eval = torch.from_numpy(np.array(y_eval)).type(torch.LongTensor)

        # train_data = my_dataset(df=X_train, labels=y_train, transform=transform)
        eval_data = my_dataset(df=X_eval, labels=y_eval, transform=transform)

        # data_loader_train = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        data_loader_eval = DataLoader(eval_data, batch_size=BATCH_SIZE, shuffle=True)





        for ind, each_model in enumerate(model_list):

            cur_model_name = each_model.split('/')[-1].replace('.pth', '')

            eval_loss = 0
            eval_acc = 0
            eval_step = 0
            using_model.load_state_dict(torch.load(each_model,map_location=torch.device(device)))
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

            if 'train' in data_path:
                if ind == 0:
                    usage_dict = {'Loss_Train:': eval_loss, 'Acc_Train':eval_acc}
                    result_df_train = pd.DataFrame(usage_dict, index=[cur_model_name])
                else:
                    usage_dict = {'Loss_Train:': eval_loss, 'Acc_Train':eval_acc}
                    result_df_train = result_df_train.append(pd.DataFrame(usage_dict, index=[cur_model_name]))


            if 'test' in data_path:
                if ind == 0:
                    usage_dict = {'Loss_Test:': eval_loss, 'Acc_Test':eval_acc}
                    result_df_test = pd.DataFrame(usage_dict, index=[cur_model_name])
                else:
                    usage_dict = {'Loss_Test:': eval_loss, 'Acc_Test':eval_acc}
                    result_df_test = result_df_test.append(pd.DataFrame(usage_dict, index=[cur_model_name]))


            print(f'Evaluation of model {cur_model_name} Acc: {eval_acc}, Loss: {eval_loss}')

    result_df = pd.concat((result_df_train, result_df_test),1)

    result_df.to_csv(result_csv_saved_path,index='Model')
    # st()
    print(f'Evaluation is done! Result CSV is saved at {result_csv_saved_path}')


if __name__ == '__main__':
    main()
