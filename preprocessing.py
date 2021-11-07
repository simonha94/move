import torch
import h5py
import numpy as np
import os


root_dir = '/home/hachmeier/move'
#root_dir = '/home/simon/Documents/repos/move'
data_dir = root_dir + '/data'
u2_test_dir = data_dir + '/u2_test'



def load_h5_to_np(file):
    """transform from h5 to np array"""
    f = h5py.File(file, 'r')
    return np.array(f['crema'])


def np_to_move_dim(np_array):
    """dimension reshaping to the standards of MOVE"""
    # this is casting the crema feature to a torch.Tensor type
    cremaPCP_tensor = torch.from_numpy(np_array).t()

    # this is the resulting cremaPCP feature
    cremaPCP_reshaped = torch.cat((cremaPCP_tensor, cremaPCP_tensor))[:23].unsqueeze(0)
    return cremaPCP_reshaped


def dataset_dict():
    pass


def preprocess():
    """here we want to loop through the files in the /u2_test directory and create the dataset_dict"""
    data = []
    labels = []
    for file in os.listdir(u2_test_dir):

        cremaPCP = np_to_move_dim(load_h5_to_np(u2_test_dir + '/' + file))  # loading the cremaPCP features for the ith song of your dataset
        #print("Shape of crema feature: {}".format(cremaPCP.shape))
        label = file[0]  # loading the label of the ith song of your dataset, indicated by filename prefix

        data.append(cremaPCP)
        labels.append(label)

    dataset_dict = {'data': data, 'labels': labels}
    #print("Labels")
    for key, value in dataset_dict.items():
    	if key == 'labels':
    		print(value)
    for dataf in data:
        print("Data.shape: {}".format(dataf.shape))
    torch.save(dataset_dict, os.path.join(root_dir, 'data', 'benchmark_crema.pt'))


def annotations():
    labels = torch.load(os.path.join(data_dir, 'benchmark_crema.pt'))['labels']

    ytrue = []

    for i in range(len(labels)):
        main_label = labels[i]  # label of the ith song
        sub_ytrue = []
        for j in range(len(labels)):
            if labels[j] == main_label and i!= j:  # checking whether the ith and jth song has the same label
                sub_ytrue.append(1)
            else:
                sub_ytrue.append(0)
        ytrue.append(sub_ytrue)

    ytrue = torch.Tensor(ytrue)
    torch.save(ytrue, os.path.join(data_dir, 'ytrue_benchmark.pt'))

if __name__ == '__main__':
	preprocess()
	annotations()
