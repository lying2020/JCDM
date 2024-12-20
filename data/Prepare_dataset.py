from PIL import Image
from torch.utils.data import Dataset
import data.util as Util
import os

class PrepareDataset(Dataset):
    def __init__(self, split, dataroot, datatype, data_len=-1):
        self.datatype = datatype
        self.data_len = data_len
        self.split = split
        gt_dir = 'gt'
        input_dir = 'input'
        mask_dir = 'mask'

        if datatype == 'img':
            clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
            noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
            mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

            self.clean_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
            self.noise_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
            self.mask_path = [os.path.join(dataroot, mask_dir, x) for x in mask_files]

            self.dataset_len = len(self.clean_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        noise_img = Image.open(self.noise_path[index]).convert("RGB")        
        clean_img = Image.open(self.clean_path[index]).convert("RGB")
        img_mask = Image.open(self.mask_path[index]).convert("1")
        [noise_img, clean_img, img_mask] = Util.transform_augment([noise_img, clean_img, img_mask], split=self.split, min_max=(-1, 1))
        return {'GT': clean_img, 'IN': noise_img, 'mask': img_mask, 'Index': index}
