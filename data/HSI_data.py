import torch
import torch.utils.data as data


class HSIdata(data.Dataset):
    def __init__(self, data_sample, cfg):
        self.phase = cfg['phase']
        self.img = data_sample['pad_img']
        self.img_indices = data_sample['pad_img_indices']
        self.gt = data_sample['img_gt']

        self.pca = 'img_pca_pad' in data_sample
        if self.pca:
            self.img_pca = data_sample['img_pca_pad']
        if self.phase == 'train':
            self.data_indices = data_sample['train_indices']
        elif self.phase == 'test':
            self.data_indices = data_sample['test_indices']
        elif self.phase == 'no_gt':
            self.data_indices = data_sample['no_gt_indices']

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        index = self.data_indices[idx]
        img_index = self.img_indices[index[0]]
        img = self.img[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
        label = self.gt[index[1], index[2]]
        if self.pca:
            img_pca = self.img_pca[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
            return img, label, index, img_pca
        else:
            return img, label, index


def batch_collate(batch):
    images = []
    labels = []
    indices = []
    images_pca = []
    for sample in batch:
        images.append(sample[0])
        labels.append(sample[1])
        indices.append(sample[2])
        if len(sample) > 3:
            images_pca.append(sample[3])
    if len(images_pca) > 0:
        return torch.stack(images, 0), torch.stack(labels), torch.stack(indices), torch.stack(images_pca, 0)
    else:
        return torch.stack(images, 0), torch.stack(labels), torch.stack(indices)
