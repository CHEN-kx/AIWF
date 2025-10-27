# The data should be stored in the npy format with the time as the name.

import os
import re
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset

class Dataset_pangu(Dataset):
    def __init__(self,
                 split="train",
                 years=range(1979,2019),
                 data_dir = '/datadir'):
        
        self.split= split
        self.data_dir = data_dir
        if split=="train":
            self.data_list = self.load_dataset([y for y in years][:-2])
        elif split=="val":
            self.data_list = self.load_dataset([y for y in years][-1:])
        elif split=="test":
            self.data_list = self.load_dataset([y for y in years][-2:-1])

    def load_dataset(self, year_list):
        ds = []
        for y in year_list:
            root_path = os.path.join(self.data_dir, str(y))
            for i in os.listdir(root_path):
                ds.append(os.path.join(str(y),i))
        ds.sort()
        return ds
    
    def __len__(self):    
        if self.split=='train':
            return len(self.data_list)-1
        elif self.split=='val':
            return len(self.data_list)-21
        else:
            return len(self.data_list)

    def __getitem__(self, idx):
        if self.split=='train':
            input_path = os.path.join(self.data_dir, self.data_list[idx])
            target_path = os.path.join(self.data_dir, self.data_list[idx+1])
            
            input_data = torch.from_numpy(np.load(input_path))
            input_air = input_data[:,:65].reshape(5,13,721,1440)
            input_surface = input_data[:,65:].reshape(4,721,1440)  

            target_data = torch.from_numpy(np.load(target_path))
            target_air = target_data[:,:65].reshape(5,13,721,1440)
            target_surface = target_data[:,65:].reshape(4,721,1440)
            return (input_air,input_surface), (target_air,target_surface)
        else:
            input_path = os.path.join(self.data_dir, self.data_list[idx])
            input_data = torch.from_numpy(np.load(input_path))
            
            input_air = input_data[:,:65].reshape(5,13,721,1440)
            input_surface = input_data[:,65:].reshape(4,721,1440)  

            target = []
            for i in range(idx+1,idx+21):
                target_path = os.path.join(self.data_dir, self.data_list[i])
                target_data = torch.from_numpy(np.load(target_path))
                target.append(target_data)
            target = torch.concat(target, dim=0)
            return (input_air,input_surface), target


if __name__=='__main__':
    # dataset=Dataset_pangu(split="train")
    # dataset[0]
    # dataset[dataset.len_data_list1-1]
    # dataset[dataset.len_data_list1]
    # dataset[len(dataset)-1]


    ''' test train ''' 
    # dataset=Dataset_pangu(split="train")
    # for i in tqdm(range(len(dataset)-5,len(dataset))):
    #     d = dataset[i]
        

    ''' test val ''' 
    dataset=Dataset_pangu(split='val')
    for i in tqdm(range(len(dataset)-5,len(dataset))):
        d = dataset[i]
        import pdb;pdb.set_trace()

    ''' test test ''' 
    # dataset=Dataset_pangu(split='test')
    # for i in tqdm(range(len(dataset))):
    #     d = dataset[i]
    #     import pdb; pdb.set_trace()
