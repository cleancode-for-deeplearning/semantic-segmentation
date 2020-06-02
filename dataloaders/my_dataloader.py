"""
    학습에 사용될 이미지 데이터 로더를 정의
Usage:

    DataLoaderSegmentation 클래스를 인스턴스화 시켜 iterable하게 사용
"""

import os
from PIL import Image
from glob import glob

from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataloaders.my_transform import random_rotation_transform, normalize, to_tensor

class DataLoaderSegmentation(data.Dataset):
    """
    torch.utils.data.Dataset 클래스를 상속 받아 iterable한 데이터 로더 구성 
    __ini__(), __getitem__(), __len__() 메소드 정의
    param:
        task: str train/val/test 중 선택
        transforms: torchvision.transforms 
        img_files: list[str] 이미지 파일 경로 리스트
        mask_files: list[str] 마스크 파일 경로 리스트
    return:
        dict {torch.tensor image, torch.tensor mask}
    """
    def __init__(self, folder_path, task):
        super(DataLoaderSegmentation, self).__init__()
        self.task = task # train / val / test
        self.transforms = transforms
        self.img_files = glob(os.path.join(folder_path, "images", self.task,"*.jpg"))
        self.mask_files= glob(os.path.join(folder_path, "masks", self.task, "*.png"))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path= self.mask_files[index]
        image = Image.open(img_path)
        mask  = Image.open(mask_path)

        return self._mytransforms({'image':image, 'mask':mask}, self.task)

    def __len__(self):
        return len(self.img_files)

    def _mytransforms(self, x:dict, task:str):
        if task == "train":            
            transforms_pipeline = transforms.Compose([
                random_rotation_transform(),
                normalize(),
                to_tensor()
            ])
        elif task == "val":
            transforms_pipeline = transforms.Compose([
                normalize(),                
                to_tensor()
            ])
        else:
            transforms_pipeline = transforms.Compose([
                normalize(),                
                to_tensor()
            ])
            
        return transforms_pipeline(x)
