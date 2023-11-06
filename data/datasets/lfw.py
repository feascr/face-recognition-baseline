import json
from pathlib import Path

from tqdm import tqdm
import cv2
from torch.utils.data.dataset import Dataset

from . import register_dataset


@register_dataset('lfw')
class LFWDataset(Dataset):
    def __init__(
        self,
        json_path,
        transforms,
    ):
        # see `prepare_data` for processing method
        with open(json_path, 'r') as f:
            self.pairs = json.load(f)
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def _read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index):
        pair = self.pairs[index]
        image1 = self._read_image(pair['image1'])
        image2 = self._read_image(pair['image2'])
        image1 = self.transforms(image=image1)['image']
        image2 = self.transforms(image=image2)['image']
        return image1, image2, pair['label']
    
    @classmethod
    def prepare_data(cls, data_dir, save_path=None):
        class_mapping = {'positive': 1, 'negative': 0}
        all_pairs = []
        def find_pairs(class_folder):
            pairs = []
            for i in tqdm(Path(data_dir, class_folder).glob('*')):
                pair = list(i.glob('*'))
                assert len(pair) == 2
                pairs.append(
                    {
                        'image1': str(pair[0]),
                        'image2': str(pair[1]),
                        'label': class_mapping[class_folder]
                    }
                )
            return pairs
        all_pairs = find_pairs('positive') + find_pairs('negative')
        if save_path is not None:
            with open(save_path, 'w') as f:
                json.dump(all_pairs, f)
        return all_pairs
