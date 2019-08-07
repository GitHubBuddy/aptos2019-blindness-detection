import torchvision.models as tvm
import torch.nn as nn
import torch.nn.functional as F
from models.modelZoo.resnet import *
import torch.nn.utils.weight_norm as weightNorm



class model_blindness(nn.Module):
    def __init__(self, num_classes=1, inchannels=3, model_name='resnet101'):
        super().__init__()
        planes = 512
        
        self.model_name = model_name

        if model_name == 'resnet101':
            self.basemodel = resnet101(pretrained=True)
            planes = 2048
        elif model_name == 'xception':
            self.basemodel = xception(True)
            planes = 2048
        elif model_name == 'inceptionv4':
            self.basemodel = inceptionv4(pretrained='imagenet')
            planes = 1536
        elif model_name == 'dpn68':
            self.basemodel = dpn68(pretrained=True)
            planes = 832
        elif model_name == 'dpn92':
            self.basemodel = dpn92(pretrained=True)
            planes = 2688
        elif model_name == "dpn98":
            self.basemodel = dpn98( pretrained=True)
            planes = 2688
        elif model_name == "dpn107":
            self.basemodel = dpn107( pretrained=True)
            planes = 2688
        elif model_name == "dpn131":
            self.basemodel = dpn131( pretrained=True)
            planes = 2688
        elif model_name == 'seresnext50':
            self.basemodel = se_resnext50_32x4d(inchannels=inchannels, pretrained='imagenet')
            planes = 2048
        elif model_name == 'seresnext101':
            self.basemodel = se_resnext101_32x4d(inchannels=inchannels, pretrained='imagenet')
            planes = 2048
        elif model_name == 'seresnet101':
            self.basemodel = se_resnet101(pretrained='imagenet',  inchannels=inchannels)
            planes = 2048
        elif model_name == 'senet154':
            self.basemodel = senet154(pretrained='imagenet', inchannels=inchannels)
            planes = 2048
        elif model_name == "seresnet152":
            self.basemodel = se_resnet152(pretrained='imagenet')
            planes = 2048
        elif model_name == 'nasnet':
            self.basemodel = nasnetalarge()
            planes = 4032
        else:
            assert False, "{} is error".format(model_name)

        self.avgpool = nn.AvgPool2d(7)
        self.lastlayer = nn.Sequential(
                 nn.BatchNorm1d(planes, eps=1e-05, momentum=0.1, affine=True,track_running_stats=True),
                 nn.Dropout(p=0.25),
                 nn.Linear(in_features=planes, out_features=planes, bias=True),  #(2048, 5)
                 nn.ReLU(),
                 nn.BatchNorm1d(planes, eps=1e-05, momentum=0.1, affine=True,track_running_stats=True),
                 nn.Dropout(p=0.5),
                 nn.Linear(in_features=planes, out_features=num_classes, bias=True))
        for layer in self.lastlayer:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.001)
                nn.init.constant_(layer.bias, 0)



    def forward(self, x, label=None):
        feat = self.basemodel(x)
        feat = self.avgpool(feat)
        flatfeat = feat.view(x.size(0), -1)
#        out = self.fc(flatfeat)
        out = self.lastlayer(flatfeat)
        return out



    def freeze_bn(self):
        for m in self.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.BatchNorm2d):
                    m[1].eval()

    def freeze(self):
        for param in self.basemodel.parameters():
            param.requires_grad = False
#        for param in self.basemodel.layer4.parameters():
#            param.requires_grad = True


    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            try:
                state_dict[key] = pretrain_state_dict[key]
            except:
                print(key)

        self.load_state_dict(state_dict)
