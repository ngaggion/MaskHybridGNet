import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 
import json
import os 

class LandmarksDataset(Dataset):
    def __init__(self, images, img_path, label_path, transform=None):
        
        self.images = images
        self.img_path = img_path
        self.label_path = label_path          
        self.transform = transform
        
        #check existence of all images
        for img in self.images:
            if not os.path.exists(os.path.join(self.img_path, img)):
                print("Image not found: ", os.path.join(self.img_path, img))
                # remove from list
                self.images = np.delete(self.images, np.where(self.images == img))
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        
        img_path = os.path.join(self.img_path, img_name)
        image = cv2.imread(img_path, 0).astype('float') / 255.0
        
        landmarks = json.load(open(os.path.join(self.label_path, img_name.replace('.png', '.json'))))
        
        matrix = []
        
        for key in landmarks.keys():
            array = np.array(landmarks[key])
            id = int(key)
            
            # Concat a column of id
            array = np.concatenate((array, np.ones((array.shape[0], 1)) * id), axis=1)
            matrix.append(array)
        
        landmarks = np.concatenate(matrix, axis=0) 
        
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
                
        h, w = image.shape[:2]
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        else:
            image = np.transpose(image, (2, 0, 1))
        
        landmarks[:,0] /= w
        landmarks[:,1] /= h 

        image = (image - image.min()) / (image.max() - image.min())
        
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float()}


class ToTensorWithSeg(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
                
        h, w = image.shape[:2]
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        else:
            image = np.transpose(image, (2, 0, 1))

        rasters = []
        organs = np.unique(landmarks[:,2])

        for organ in organs:
            organ_landmarks = landmarks[landmarks[:,2] == organ][:,:2]
            raster = cv2.drawContours(np.zeros((h, w)), [organ_landmarks.astype('int')], -1, 1, -1)
            rasters.append(raster)
        
        rasters = np.stack(rasters, axis = 0)
        landmarks[:,0] /= w
        landmarks[:,1] /= h 
        
        image = (image - image.min()) / (image.max() - image.min())

        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float(),
                'raster': torch.from_numpy(rasters).float()}


class ToTensorWithSegBatched(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
                
        h, w = image.shape[:2]
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        else:
            image = np.transpose(image, (2, 0, 1))

        rasters = []
        organs, counts = np.unique(landmarks[:,2], return_counts=True)

        for organ in organs:
            organ_landmarks = landmarks[landmarks[:,2] == organ][:,:2]
            raster = cv2.drawContours(np.zeros((h, w)), [organ_landmarks.astype('int')], -1, 1, -1)
            rasters.append(raster)
        
        rasters = np.stack(rasters, axis = 0)
        landmarks[:,0] /= w
        landmarks[:,1] /= h 
        
        landmarks = torch.from_numpy(landmarks).float()
        # Hardcode max contour size
        landmarks_batched = torch.zeros(counts.shape[0], 5000, 2)
        idx = 0
        for i, organ in enumerate(organs):
            organ_landmarks = landmarks[landmarks[:,2] == organ, :2]
            landmarks_batched[idx, :counts[i], :] = organ_landmarks
            idx += 1
        
        image = (image - image.min()) / (image.max() - image.min())

        return {'image': torch.from_numpy(image).float(),
                'landmarks': landmarks_batched.float(),
                'organs': torch.from_numpy(organs).long(),
                'counts': torch.from_numpy(counts).long(),
                'raster': torch.from_numpy(rasters).float()}
    



class TestDataset(Dataset):
    def __init__(self, images, img_path, transform=None):
        
        self.images = images
        self.img_path = img_path        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        
        if self.img_path is not None:
            img_path = os.path.join(self.img_path, img_name)
        else:
            img_path = img_name
        image = cv2.imread(img_path, 0).astype('float') / 255.0
                
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class ToTensorTest(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
                
        h, w = image.shape[:2]
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        else:
            image = np.transpose(image, (2, 0, 1))
        
        image = (image - image.min()) / (image.max() - image.min())
        
        return {'image': torch.from_numpy(image).float()}
    

