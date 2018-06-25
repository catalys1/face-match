import torch
import model


def tocuda(seq):
    if torch.cuda.is_available():
        return [x.cuda() for x in seq]
    return seq


def load_model(checkpoint=None):
    print('Loading network...')
    net = model.FaceNet()
    if torch.cuda.is_available():
        net = net.cuda()
    if checkpoint:
        if torch.cuda.is_available():
            params = torch.load(checkpoint)
        else:
            params = torch.load(checkpoint, map_location=lambda s, l: s)
        net.load_state_dict(params)
    return net


