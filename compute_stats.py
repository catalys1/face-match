import torch
import numpy as np
from matplotlib import pyplot as plt
import tqdm
import utils
import dataset


def pair_distance(net, dataloader):
    net.eval()
    dist = []
    labs = []
    for batch in tqdm.tqdm(dataloader):
        img1, img2 = utils.tocuda(batch[:-1])
        label = batch[-1]
        with torch.no_grad():
            emb1 = net(img1)
            emb2 = net(img2)

            d = torch.pow(emb1 - emb2, 2).sum(1)
            dist += d.cpu().tolist()
            labs += label.tolist()

    dist = np.array(dist)
    labs = np.array(labs)
    return dist, labs


def roc_curve(dist, labs):
    srt = dist.argsort()
    dist = dist[srt]
    labs = labs[srt]

    same = dist[labs == 0]
    diff = dist[labs == 1]

    roc_points = []
    accuracy = []
    for d in dist:
        cp = same <= d
        cn = diff > d
        tpr = np.mean(cp)
        fpr = np.mean(1 - cn)
        acc = (cp.sum() + cn.sum()) / (cp.size + cn.size)
        
        roc_points.append([fpr, tpr])
        accuracy.append([d, acc])

    roc_points = np.array(roc_points).T
    accuracy = np.array(accuracy).T
    return roc_points, accuracy


def main(args):
    data = dataset.RandomTripletDataset(train=False)
    loader = torch.utils.data.DataLoader(data, batch_size=20, num_workers=10)
    net = utils.load_model(args.model)
    pair_dist, labels = pair_distance(net, loader)
    roc, acc = roc_curve(pair_dist, labels)
    open('roc.txt', 'w').write(
        '\n'.join(','.join(str(x) for x in p) for p in roc.T))
    if args.plot:
        plt.subplot(211)
        plt.plot(roc[0], roc[1], label='roc')
        plt.grid()
        plt.legend()
        plt.subplot(212)
        plt.plot(acc[0], acc[1], label='accuracy')
        plt.yticks(np.arange(0, 1.001, 0.05))
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to the model that will be evaluated')
    parser.add_argument('--plot', action='store_true', help='Plot ROC curve')

    args = parser.parse_args()
    main(args)
