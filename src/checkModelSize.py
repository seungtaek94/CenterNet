from lib.opts import opts
from lib.models.model import create_model, load_model
from types import MethodType
import torch.onnx as onnx
import torch

from torch.onnx import OperatorExportTypes
from collections import OrderedDict
from torchsummary import summary

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")

## onnx is not support dict return value
## for dla34
def pose_dla_forward(self, x):
    x = self.base(x)
    x = self.dla_up(x)
    y = []
    for i in range(self.last_level - self.first_level):
        y.append(x[i].clone())
    self.ida_up(y, 0, len(y))
    ret = []  ## change dict to list
    for head in self.heads:
        ret.append(self.__getattr__(head)(y[-1]))
    return ret
## for dla34v0
def dlav0_forward(self, x):
    x = self.base(x)
    x = self.dla_up(x[self.first_level:])
    # x = self.fc(x)
    # y = self.softmax(self.up(x))
    ret = []  ## change dict to list
    for head in self.heads:
        ret.append(self.__getattr__(head)(x))
    return ret
## for resdcn
def resnet_dcn_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.deconv_layers(x)
    ret = []  ## change dict to list
    for head in self.heads:
        ret.append(self.__getattr__(head)(x))
    return ret

forward = {'dla':pose_dla_forward,'dlav0':dlav0_forward,'resdcn':resnet_dcn_forward}

opt = opts().init()  ## change lib/opts.py add_argument('task', default='ctdet'....) to add_argument('--task', default='ctdet'....)
opt.arch = 'dla_34'
opt.heads = OrderedDict([('hm', 8), ('reg', 2), ('wh', 2)])
opt.head_conv = 256 if 'dla' in opt.arch else 64
print(opt)
model = create_model(opt.arch, opt.heads, opt.head_conv)
model.forward = MethodType(forward[opt.arch.split('_')[0]], model)


#model.cuda()
#summary(model, (3, 416, 416))

from ptflops import get_model_complexity_info
#

#with torch.cuda.device(0):
net = model.cuda()
macs, params = get_model_complexity_info(net, (3, 416, 416), as_strings=True, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
'''
load_model(model, '../models/ctdet_coco_dla_2x.pth')
model.eval()
model.cuda()
input = torch.zeros([1, 3, 416, 416]).cuda()
onnx.export(model, input, "ctdet_cocoAxle_dla_2x.onnx", verbose=True,
            operator_export_type=OperatorExportTypes.ONNX)
'''