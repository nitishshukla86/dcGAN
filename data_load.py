import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
class FRLL(Dataset):

    def __init__(self,train=True, transform=None,dataset='amsl',file_path=''):
        self.df=pd.read_csv(file_path)
        self.df=self.df[self.df['dataset']==dataset]
        self.df=self.df[self.df['is_train']==int(train)]
        self.transform = transform 
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row=list(self.df.iloc[idx])
        image=cv2.cvtColor(cv2.imread(row[-3]), cv2.COLOR_BGR2RGB)
        img1=cv2.cvtColor(cv2.imread(row[-2]), cv2.COLOR_BGR2RGB)
        img2=cv2.cvtColor(cv2.imread(row[-1]), cv2.COLOR_BGR2RGB)
        # print(img1.shape,img2.shape)
        if self.transform:
            image=self.transform(image)
            img1=self.transform(img1)
            img2=self.transform(img2)
        sample = {'morphed_image': image, 'img1': img1, 'img2':img2}
        return sample


