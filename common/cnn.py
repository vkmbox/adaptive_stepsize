import torch
from torch import nn
from torchvision.models import resnet18

def simple_cnn(in_channels, num_classes, dropout_const = 1.0):
    
    model = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3), padding='same'),
        nn.LeakyReLU(0.1),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same'),
        nn.LeakyReLU(0.1),
        nn.MaxPool2d(kernel_size = (2,2)),
        nn.Dropout(0.25*dropout_const),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same'),
        nn.LeakyReLU(0.1),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same'),
        nn.LeakyReLU(0.1),
        nn.MaxPool2d(kernel_size = (2,2)),
        nn.Dropout(0.25*dropout_const),
        nn.Flatten(),
        nn.Linear(4096, 256),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.5*dropout_const),
        nn.Linear(256, num_classes)
    )

    return model
    
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def make_resnet9(in_channels, num_classes):
  return ResNet9(in_channels, num_classes)

def shortcut_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        self.res2 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        self.res3 = nn.Sequential(conv_block(64, 128, pool=True), conv_block(128, 128))
        self.short3 = shortcut_block(64, 128)
        self.res4 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.res5 = nn.Sequential(conv_block(128, 256, pool=True), conv_block(256, 256))
        self.short5 = shortcut_block(128, 256)
        self.res6 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.res7 = nn.Sequential(conv_block(256, 512, pool=True), conv_block(512, 512))
        self.short7 = shortcut_block(256, 512)
        self.res8 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        #out = self.conv2(out)
        out = self.res1(out) + out
        out = self.res2(out) + out
        #raise Exception("out dim:{}, res3 dim:{}, short3 dim:{}"\
        #    .format(out.shape, self.res3(out).shape, self.short3(out).shape))
        out = self.res3(out) + self.short3(out)
        out = self.res4(out) + out
        out = self.res5(out) + self.short5(out)
        out = self.res6(out) + out
        out = self.res7(out) + self.short7(out)
        out = self.res8(out) + out
        out = self.classifier(out)
        return out

def make_resnet18(in_channels, num_classes):
    return ResNet18(in_channels=in_channels, num_classes=num_classes)

def modify_torchvision_resnet(resnet):
    '''modify torchvision's resnet18 to better suit cifar10'''
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    return resnet

def make_resnet18v2(in_channels, num_classes):
    model = resnet18(num_classes=num_classes)
    return modify_torchvision_resnet(model)
