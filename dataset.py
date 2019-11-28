import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


class MaskDataset(object):
    def __init__(self, root):
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        self.masks = []
        self.imgs = []
        for mask_file in masks:
            img_mask_path = os.path.join(root, 'mask', mask_file)
            img_file = mask_file.replace('.mask.', '.sat.').replace('.png', '.jpg')
            img_mask = Image.open(img_mask_path).quantize(colors=256, method=2)
            img_mask = np.array(img_mask)
            if np.min(img_mask) == np.max(img_mask):
                continue

            self.masks.append(mask_file)
            self.imgs.append(img_file)

        print('loaded: tiles = {} x masks = {}'.format(len(self.imgs), len(self.masks)))

        # self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "tile"))))

    @staticmethod
    def _normalize_min_max(min_, max_):
        if min_ == max_:
            if max_ == 255:
                min_ -= 1
            else:
                max_ += 1
        elif min_ > max_:
            min_, max_ = max_, min_

        return min_, max_

    def __getitem__(self, idx):
        # load images ad masks
        idx = idx % len(self.imgs)

        img_path = os.path.join(self.root, "tile", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])

        # img = Image.open(img_path).convert("RGB")

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        img_mask = Image.open(mask_path).quantize(colors=256, method=2)

        # img_mask = img_mask.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)
        # alpha = img_mask.split()[-1]
        # the_mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
        # img_mask.paste(255, the_mask)

        # convert the PIL Image into a numpy array
        img_mask = np.array(img_mask)

        # instances are encoded as different colors
        obj_ids = np.unique(img_mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = img_mask == obj_ids[:, None, None]
        masks = np.bitwise_not(masks)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        try:
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])

                xmin, xmax = self._normalize_min_max(xmin, xmax)
                ymin, ymax = self._normalize_min_max(ymin, ymax)

                boxes.append([xmin, ymin, xmax, ymax])

        except IndexError as e:
            print(e)
            print(img_path)
            print(mask_path)
            raise

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        if boxes.size()[0] > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.as_tensor(0)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["area"] = area
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        transforms = self.get_transform()
        img_tensor = transforms(Image.open(img_path).convert("RGB"))

        return img_tensor, target

    def __len__(self):
        return len(self.imgs)

    def get_transform(self):
        transforms = list()
        # if self.is_train:
        #     transforms.append(T.RandomHorizontalFlip(0.5))

        # if self.is_resize:
        #     transforms.append(T.Resize((224, 224)))

        transforms.append(T.ToTensor())
        # transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        return T.Compose(transforms)
