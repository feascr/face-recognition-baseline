import json
import random
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset

from . import register_dataset

@register_dataset('web_face')
class WebFaceDataset(Dataset):
    def __init__(
        self,
        json_path,
        augmentations,
    ):
        # see `prepare_data` for processing method
        with open(json_path, 'r') as f:
            self.procesed_data = {int(k): v for k, v in json.load(f).items()}
        self.augmentations = augmentations

    def __len__(self):
        return len(self.procesed_data)

    def _read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index):
        user_image_path = random.choice(self.procesed_data[index])
        user_image = self._read_image(user_image_path)
        user_image = self.augmentations(image=user_image)['image']
        return user_image, index
    
    def get_info_for_sampler(self):
        weights = np.asarray([1.0] * len(self.procesed_data))
        weights = weights / len(self.procesed_data)
        return weights
    
    @classmethod
    def prepare_data(cls, data_dir, save_path=None):
        user2img = defaultdict(list)
        for i in tqdm(Path(data_dir).rglob('*')):
            if i.is_file():
                user_id = i.parent.parts[-1].split('_')[1]
                user2img[int(user_id)].append(str(i))
        if save_path is not None:
            with open(save_path, 'w') as f:
                json.dump(user2img, f)
        return user_id
