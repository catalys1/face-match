import torch
from torchvision import transforms as T
from pathlib import Path
import json
import PIL.Image as Image
import random


def augmentation():
    a = T.Compose([
        T.RandomCrop(224),
        T.ColorJitter(0.2, 0.2, 0.2),
    ])
    return a


class RandomTripletDataset(torch.utils.data.Dataset):
    
    def __init__(self, root='../lfw/', train=True, augment=False):
        super(RandomTripletDataset, self).__init__()

        root = Path(root)
        self.root = root
        self.images = self.root.joinpath('images/')
        self.train = train
        
        freqs = json.load(open(root.joinpath('frequencies.json')))['freq']
        split = json.load(open(root.joinpath('train_test.json')))
        if train:
            self.data = split['train']
            self.positives = sorted([k for k in self.data if freqs[k] > 1])
        else:
            self.data = split['test']
            self.pairs = json.load(
                open(root.joinpath('pairs_dev_test.json')))
            self.pairs = self.pairs['same'] + self.pairs['diff']

        self.toid = dict(zip(self.data.keys(), range(len(self.data))))
        self.fromid = dict(zip(range(len(self.data)), self.data.keys()))
        self.freqs = freqs

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.tensor_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)])

        if augment:
            self.augment = augmentation()
        else:
            self.augment = None


    def __len__(self):
        if self.train:
            return len(self.positives)
        else:
            return len(self.pairs)
        

    def _get_train(self, index):
        person = self.positives[index]
        i1, i2 = random.sample(range(self.freqs[person]), 2)
        anchor = Image.open(self.images.joinpath(self.data[person][i1]))
        pos = Image.open(self.images.joinpath(self.data[person][i2]))

        p1, p2 = random.sample(self.data.keys(), 2)
        neg_person = p1 if p1 != person else p2
        i3 = random.choice(range(self.freqs[neg_person]))
        neg = Image.open(self.images.joinpath(self.data[neg_person][i3]))

        if self.augment:
            anchor = self.augment(anchor)
            pos = self.augment(pos)
            neg = self.augment(neg)

        anchor = self.tensor_normalize(anchor.resize((224,224), Image.BILINEAR))
        pos = self.tensor_normalize(pos.resize((224,224), Image.BILINEAR))
        neg = self.tensor_normalize(neg.resize((224,224), Image.BILINEAR))
        ind = torch.LongTensor([
            [self.toid[person], self.toid[person], self.toid[neg_person]], 
            [i1, i2, i3]])

        return anchor, pos, neg, ind


    def _get_test(self, index):
        pair = self.pairs[index]
        # 0 if same, 1 if different
        label = 0 if index < 500 else 1

        img1 = Image.open(self.images.joinpath(pair[0]))
        img2 = Image.open(self.images.joinpath(pair[1]))
        img1 = self.tensor_normalize(img1.resize((224,224), Image.BILINEAR))
        img2 = self.tensor_normalize(img2.resize((224,224), Image.BILINEAR))

        return img1, img2, label


    def __getitem__(self, index):
        if self.train:
            return self._get_train(index)
        else:
            return self._get_test(index)

