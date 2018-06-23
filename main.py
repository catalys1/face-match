import torch
import numpy as np
import dataset
import time
from logger import Logger
import utils


class TripletLoss(torch.nn.Module):

    def __init__(self, a=0.2):
        super(TripletLoss, self).__init__()
        self.a = a
    
    def forward(self, anchor, positive, negative):
        dp = torch.pow(anchor - positive, 2).sum(1)
        dn = torch.pow(anchor - negative, 2).sum(1)
        loss = torch.mean(torch.clamp(dp - dn + self.a, min=0.0))
        return loss


class HardTripletMiningLoss(torch.nn.Module):
    
    def __init__(self, a=0.2):
        super(HardTripletMiningLoss, self).__init__()
        self.a = a

    def forward(self, anchor, positive, negative, ind):
        embeddings = torch.cat([anchor, positive, negative], 0)
        labels = ind[:, 0].t().contiguous().view(-1).cuda()

        pair_dist = torch.mm(embeddings, embeddings.t())
        sq_norm = pair_dist.diag()
        pair_dist = sq_norm.view(1, -1) + sq_norm.view(-1, 1) - 2 * pair_dist
        pair_dist = pair_dist.clamp_(min=0)

        label_same = labels.view(-1, 1).eq(labels)
        label_diff = 1 - label_same

        triplet_diff = pair_dist.unsqueeze(2) - pair_dist + self.a
        triplet_diff[label_diff] = 0
        triplet_diff[:, label_same] = 0
        triplet_diff[torch.diag(torch.zeros(
            triplet_diff.size()[0], dtype=torch.long))] = 0
        hard = triplet_diff[triplet_diff > 0]
        if hard.nelement() > 0:
            loss = hard.mean()
        else:
            loss = 0
        return loss


def train(loader, net, loss_fn, opt, epoch):
    N = int(np.ceil(len(loader.dataset) / loader.batch_size))
    loss = 0.0
    violations = 0
    total = 0
    net.train()
    i = 0
    for i, batch in enumerate(loader):
        opt.zero_grad()
        anchor, pos, neg = utils.tocuda(batch[:-1])
        ind = batch[-1]
        anchor.requires_grad_()
        pos.requires_grad_()
        neg.requires_grad_()

        anc_embed = net(anchor)
        pos_embed = net(pos)
        neg_embed = net(neg)

        #batch_loss = loss_fn(anc_embed, pos_embed, neg_embed)
        batch_loss = loss_fn(anc_embed, pos_embed, neg_embed, ind)
        if batch_loss > 0:
            batch_loss.backward()
            opt.step()
            loss += batch_loss.item()

       #v = (torch.pow(anc_embed - pos_embed, 2).sum(1) + 0.2 >
       #    torch.pow(anc_embed - neg_embed, 2).sum(1))
       #violations += v.sum().item()
       #total += anchor.size()[0]
        print(f'\rEPOCH {epoch}: train batch {i:04d}/{N}{" "*10}',
              end='', flush=True)
    loss /= (i + 1)

    opt.zero_grad()
    return loss


def test(loader, net, epoch):
    N = int(np.ceil(len(loader.dataset) / loader.batch_size))
    net.eval()
    dist = []
    labs = []
    for i, batch in enumerate(loader):
        img1, img2 = utils.tocuda(batch[:-1])
        label = batch[-1]

        with torch.no_grad():
            img1_embed = net(img1)
            img2_embed = net(img2)

            d = torch.pow(img1_embed - img2_embed, 2).sum(1)
            dist += d.cpu().tolist()
            labs += label.cpu().tolist()

        print(f'\rEPOCH {epoch}: test batch {i:04d}/{N}{" "*10}',
              end='', flush=True)

    dist = np.array(dist)
    labs = np.array(labs)
    meds = np.median(dist.reshape(2, -1), axis=1)
    threshold = meds[0] + (meds[1] - meds[0]) / 2
    thresh = dist > threshold
    accuracy = np.mean(thresh == labs)
    return accuracy


def train_loop(args):

    print('Creating data loaders...')
    train_data = dataset.RandomTripletDataset(train=True)
    val_data = dataset.RandomTripletDataset(train=False)
    train_load = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=25, num_workers=15)
    val_load = torch.utils.data.DataLoader(
        val_data, shuffle=False, batch_size=25, num_workers=15)

    net = utils.load_model(args.model)

    #loss_fn = TripletLoss()
    loss_fn = HardTripletMiningLoss(a=0.2)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    print('Starting training')
    #log = Logger('logs/stats')
    best_val = 0
    epochs = args.epochs
    for e in range(args.start, epochs):
        start_time = time.time()
        train_loss = train(
            train_load, net, loss_fn, opt, e)
        test_acc = test(
            val_load, net, e)
        t = time.time() - start_time
        s = (f'\rEPOCH {e} (lr={opt.param_groups[0]["lr"]:1.0e}, time={t:.1f}): '
             f'Train [{train_loss:.4f}]  '
             f'Val [{test_acc * 100:02.2f}]  ')
        print(s)
        mode = 'w' if e == 0 else 'a'
        open('logs/stats.txt', mode).write(s[1:] + '\n')
        #log.log(np.array(
        #    [train_loss, train_acc, test_loss, test_acc], dtype=np.float32))

        if test_acc > best_val:
            best_val = test_acc
            torch.save(net.state_dict(), 'checkpoints/best_model')
        if e % args.interval == 0:
            torch.save(net.state_dict(), f'checkpoints/model_{e}_epochs')
 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False,
        help='Run inference on the model')
    parser.add_argument('--start', type=int, default=0,
        help='Epoch to resume from')
    parser.add_argument('--epochs', type=int, default=50,
        help='Epoch to end at')
    parser.add_argument('--lr', type=float, default=1e-3, 
        help='Learning rate')
    parser.add_argument('--model', default='',
        help='Model checkpoint to load')
    parser.add_argument('--interval', type=int, default=5,
        help='Save model state every interval epochs')
    args = parser.parse_args()

    if not args.test:
        train_loop(args)
    else:
        run_inference(args)

