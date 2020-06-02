"""
    입력 이미지 데이터를 transform할 때 사용할 trasforms 클래스 정의
Usage:

    torch.utils.data.Dataset 클래스를 상속 받은 클래스 내부에서 사용
    torchvision.transforms.Compose()에 class 명 입력
"""

import random
import numpy as np

import torch
import torchvision.transforms.functional as TF

class random_rotation_transform:
    """
    입력 이미지 데이터들을 가져와 random한 각도로 회전 시킵니다.
    param: 
        max_angle: 최대 회전 각도
        x: dict {PIL image, PIL mask}
    return: 
        dict {PIL image, PIL mask}
    """
    def __init__(self, max_angle=10):
        self.max_angle = max_angle
        
    def __call__(self, x):
        image = x['image']
        mask = x['mask']
        if random.random() > 0.5:
            angle = random.randint(-self.max_angle, self.max_angle)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        return {'image':image, 'mask':mask}
    
class normalize:
    """
    입력 이미지 데이터들을 가져와 노멀라이즈 시킵니다.
    param:
        x: dict {PIL image, PIL mask}
    return: 
        dict {np.array image, np.array mask}
    """
    def __call__(self, x):
        image = x['image']
        mask = x['mask']
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        image = image / 255.
        mask = mask / 255.
        
        return {'image':image, 'mask':mask}
    
class to_tensor:
    """
    입력 이미지 데이터를 가져와 torch tensor로 치환합니다.
    param:
        x: dict {PIL image, PIL mask}
    return:
        dict {torch.tensor image, torch.tensor mask}
    """
    def __call__(self, x):
        image = x['image']
        mask = x['mask']
        
        image = np.array(image, dtype=np.float32).transpose((2, 0, 1))
        mask = np.array(mask, dtype=np.float32)
        
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        return {'image':image, 'mask':mask}