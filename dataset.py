import numpy as np
import torch, os, cv2


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, folder, img_h, img_w, num_classes=1, transform=None):
        self.imgs_dir = f"{folder}/imgs/"
        self.masks_dir = f"{folder}/masks/"
        self.img_h = img_h
        self.img_w = img_w
        self.transform = transform
        self.num_classes = num_classes

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.imgs_dir) if not file.startswith(".")]
        print(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    @classmethod
    def pre_process(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img / 255
        return img

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = f"{self.masks_dir}{idx}.png"
        img_file = f"{self.imgs_dir}{idx}.png"

        img = np.array(cv2.cvtColor(cv2.imread(img_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        mask = np.array(cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE))

        if self.transform is None:
            img = cv2.resize(img, (self.img_w, self.img_h))
            mask = cv2.resize(mask, (self.img_w, self.img_h))
        else:
            augmented = self.transform(**{"image": img, "mask": mask})
            img, mask = augmented["image"], augmented["mask"]

        # HWC to CHW
        img = self.pre_process(img)
        img_t = torch.from_numpy(img).type(torch.FloatTensor)

        if self.num_classes > 1:
            mask_t = torch.from_numpy(mask).type(torch.LongTensor)
        else:
            mask[mask != 0] = 1
            mask = self.pre_process(mask)
            mask_t = torch.from_numpy(mask).type(torch.FloatTensor)

        return img_t, mask_t
