import torch
import model


def tocuda(seq):
    return [x.cuda() for x in seq]


def load_model(checkpoint=None):
    print('Loading network...')
    net = model.FaceNet().cuda()
    if checkpoint:
        net.load_state_dict(torch.load(checkpoint))
    return net


