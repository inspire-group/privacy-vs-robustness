from collections import OrderedDict
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, width_num=16):
        super(VGG, self).__init__()

        self.num_channels = 1
        self.num_labels = 10
        self.width_num = width_num
        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 16*self.width_num, 3, stride=1, padding=1)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(16*self.width_num, 16*self.width_num, 3, stride=1, padding=1)),
            ('relu2', activ),
            #('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(16*self.width_num, 16*self.width_num, 3, stride=2, padding=1)),
            ('relu3', activ),            
            ('conv4', nn.Conv2d(16*self.width_num, 32*self.width_num, 3, stride=1, padding=1)),
            ('relu4', activ),
            #('maxpool2', nn.MaxPool2d(2, 2)),
            ('conv5', nn.Conv2d(32*self.width_num, 32*self.width_num, 3, stride=1, padding=1)),
            ('relu5', activ),
            ('conv6', nn.Conv2d(32*self.width_num, 32*self.width_num, 3, stride=2, padding=1)),
            ('relu6', activ),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(32*self.width_num*7*7, 200)),
            ('relu1', activ),
            ('fc2', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc2.weight, 0)
        nn.init.constant_(self.classifier.fc2.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 32*self.width_num*7*7))
        return logits