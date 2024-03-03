import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils.dropout import dropout_node
from functools import partial

sys.path.append("model/")
from graph_utils import feature_map_to_graph

sys.path.append("model/learn_poly_sampling")
from layers import get_logits_model, get_antialias, PolyphaseInvariantDown2D, LPS, PolyphaseInvariantUp2D, LPS_u
from layers.polydown import set_pool
from layers.polyup import set_unpool


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='circular', bias=False),#zeros
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class LPS_Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lpd = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=in_channels,h_ch=out_channels) #p_ch: in_channel, h_ch: hidden_channel

    def forward(self, x, ret_prob=False):
        return self.lpd(x, ret_prob=ret_prob)

'''
graph module
'''

class spaG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gconv1 = SAGEConv(in_channels, in_channels)
        self.gconv2 = SAGEConv(in_channels, out_channels)
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        print(f"FM {x.shape}")
        graph = feature_map_to_graph(x)
        gfm = self.gconv1(graph.x, graph.edge_index)
        print(f"first gfm {gfm.shape}")
        gfm = gfm.relu()
        gfm = self.gconv2(gfm, graph.edge_index)
        print(f"second gfm {gfm.shape}")
        gresized =  gfm.view(batch_size, channels, height, width) # Reverse the flattening of node features
        print(f"gresized {gresized.shape}")
        return nn.Upsample(size=(600, 1200), mode='bilinear')(gresized)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class LPS_Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        antialias_layer = get_antialias(antialias_mode='LowPassFilter',
                                    antialias_size=3,
                                    antialias_padding='same',
                                    antialias_padding_mode='circular',
                                    antialias_group=1)
        self.lpu = set_unpool(partial(
            PolyphaseInvariantUp2D,
            component_selection=LPS_u,
            antialias_layer=antialias_layer), p_ch=in_channels)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2, prob):
        x1 = self.lpu(x1, prob=prob)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class SUNet_GNN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SUNet_GNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = LPS_Down(64, 64)
        self.conv1 = DoubleConv(64, 128)
        self.down2 = LPS_Down(128, 128)
        self.conv2 = DoubleConv(128, 256)
        self.down3 = LPS_Down(256, 256)
        self.conv3 = DoubleConv(256, 512)
        self.down4 = LPS_Down(512, 512)
        self.conv4 = DoubleConv(512, 1024)
        self.gconv = spaG(1024, n_classes)
          
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)
        x4 = self.down3(x3)
        x4 = self.conv3(x3)
        x5, prob = self.down4(x4, ret_prob=True)
        x5 = self.conv4(x4)
        logits = self.gconv(x5)
        return logits

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.gconv = spaG(1024, 1024)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outcov = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.gconv(x5)
        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outcov(x)
        return logits
    

class SUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = LPS_Down(64, 64)
        self.conv1 = DoubleConv(64, 128)
        self.down2 = LPS_Down(128, 128)
        self.conv2 = DoubleConv(128, 256)
        self.down3 = LPS_Down(256, 256)
        self.conv3 = DoubleConv(256, 512)
        self.down4 = LPS_Down(512, 512)
        factor = 2 if bilinear else 1
        self.conv4 = DoubleConv(512, 1024 // factor)
        self.gconv = spaG(1024, 1024)
        # self.conv5 = DoubleConv(1024, 1024)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # self.up1 = LPS_Up(1024, 512)
        # self.up2 = LPS_Up(512, 256)
        # self.up3 = LPS_Up(256, 128)
        # self.up4 = LPS_Up(128, 64)
        
        self.outcov = OutConv(64, n_classes)
          
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)
        x4 = self.down3(x3)
        x4 = self.conv3(x3)
        x5, prob = self.down4(x4, ret_prob=True)
        x5 = self.conv4(x4)
        
        x6 = self.gconv(x5)
        # x7 = self.conv5(x6)
        
        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outcov(x)
        return logits