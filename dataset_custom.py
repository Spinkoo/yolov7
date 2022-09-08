from Coco.Coco_dataset import CocoDataset, my_collate
from custom.custom_transform import RandomHorizontalFlip, RandomVerticalFlip, \
                             RandomImageCrop
import torch
import random
from  torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., prob = 0.3):
        self.std = std
        self.mean = mean
        self.prob = prob
        
    def __call__(self, tensor, target):
        if random.random() < self.prob:
          return tensor + torch.randn(tensor.size()).cuda() * self.std + self.mean, target
        return tensor, target
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_dataloader_set(path = "/content/drive/MyDrive/s/annotations/6_classes.json"):
  cc1 = CocoDataset()
  cc2 = CocoDataset()
  cc3 = CocoDataset()

  D4_transforms = [
                  # D4 Group augmentations
                  RandomHorizontalFlip(0.3),
                  RandomVerticalFlip(0.3),
                  #RandomImageCrop(0.1),
                  #trans,
                  AddGaussianNoise()
                  #A.Normalize()
                  ]
                
  cc1.load_coco(path,'', transforms = D4_transforms)
  #cc2.load_coco("drive/MyDrive/j/annotations/6_classes2.json",'', transforms = None, ignore_ids = [262, 1166, 194, 163, 1158, 868, 547, 97, 965])
  #cc3.load_coco("drive/MyDrive/truckset3.json",'', transforms = D4_transforms)

  test_set_lenght = 200

  train, test = torch.utils.data.random_split(cc1, [len(cc1) - test_set_lenght, test_set_lenght])
  #train2, test2 = torch.utils.data.random_split(cc2, [len(cc2) - test_set_lenght, test_set_lenght])
  #train3, test3 = torch.utils.data.random_split(cc3, [len(cc3) - test_set_lenght, test_set_lenght])


  #train = ConcatDataset([train1,train2, train3])
  #test = ConcatDataset([test1, test2, test3])

  train_dataloader = DataLoader(train, batch_size=6, shuffle=True, collate_fn= my_collate)

  test_dataloader = DataLoader(test, batch_size=6, shuffle=True, collate_fn= my_collate)
  return train_dataloader, test_dataloader, cc1