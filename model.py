import torch
import torchvision


class FaceNet(torch.nn.Module):
    
    def __init__(self):
        super(FaceNet, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)

        embedding = 128
        feats = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(feats, embedding)


    def forward(self, x):
        x = self.resnet(x)
        x = x / x.norm(2, 1, True)
        return x
