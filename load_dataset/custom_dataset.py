import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.io as io
import torchvision.transforms as T

# TODO: adapt this to the dataset
class HealthyLeafDataset(Dataset):
    def __init__(self, setname):
        """Tiny Dataset for 32 x 32 images for color classification.
        Variables:
       <setname> can be any of: 'train' to specify the training set
                                'val' to specify the validation set
                                'test' to specify the test set"""
        self.setname = setname
        assert setname in ['train','test']
        
        # define dataset
        overall_dataset_dir = os.path.join(os.path.join(os.getcwd(),'load_dataset'), 'dataset')
        self.selected_dataset_dir = os.path.join(overall_dataset_dir, setname)
        
        # e.g., self.all_filenames = ['img0.jpg','img1.jpg','img2.jpg']
        self.all_filenames = os.listdir(self.selected_dataset_dir)

        if setname == 'train':
            self.all_labels = pd.read_csv(os.path.join(overall_dataset_dir,'train_labels.csv'), header=0, index_col=0)
        elif setname == 'test':
            self.all_labels = pd.read_csv(os.path.join(overall_dataset_dir,'test_labels.csv'), header=0, index_col=0)
        self.label_meanings = self.all_labels.columns.values.tolist()
    
    def __len__(self):
        """Return the total number of examples in this split, e.g. if
        self.setname=='train' then return the total number of examples
        in the training set"""
        return len(self.all_filenames)
        
    def __getitem__(self, idx):
        """Return the example at index [idx]. The example is a dict with keys
        'data' (value: Tensor for an RGB image) and 'label' (value: multi-hot
        vector as Torch tensor of gr truth class labels)."""
        selected_filename = self.all_filenames[idx]
        img = io.read_image(os.path.join(self.selected_dataset_dir, selected_filename), mode=io.ImageReadMode.RGB)
        
        # normalize image to ImageNet mean and std dev (TODO: might change depending on whether using ImageNet model or not)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # required for ImageNet
        img = normalize(img)
        
        # load label
        label = torch.Tensor(self.all_labels.loc[selected_filename,:].values)
        
        sample = {'data':img, #preprocessed image, for input into NN
                  'label':label,
                  'img_idx':idx}
        return sample