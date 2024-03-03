import torch
from model.unet import UNet, SUNet
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import colors
color = colors.LinearSegmentedColormap.from_list("", ['white', 'red', 'green', 'yellow', 'blue'])

'''
load model
'''

def load_model(variant, channel, model_path):
    # initialise model
    model = UNet(n_channels=channel, n_classes=5, bilinear=False) if variant == 'basic' else SUNet(n_channels=channel, n_classes=5, bilinear=False)
    # load the weight of the trained model
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # switch the model to inference model which freeze the weight of the layer
    model.eval();
    return model

model = load_model(variant='improved', channel=3, model_path='checkpoint/improved-4/loss=0.3785_dice=0.6406.pth')

'''
read image to get layer-wise prediction
'''

image = Image.open('data/test_images/f_01340.png').convert('RGB')

img_ndarray = np.array(image)
img_ndarray = img_ndarray[np.newaxis, ...] if img_ndarray.ndim == 2 else img_ndarray.transpose((2, 0, 1))
img_ndarray = img_ndarray / 255
image_tensor = torch.from_numpy(img_ndarray).unsqueeze(0).float()

model_children = list(model.children())

with open('conv_layer/improved_conv_layer.txt', 'w') as f:
    for line in model_children:
        f.write(f"{line}\n")
        
outputs = [model_children[0](image_tensor)]

for i in range(1, 5):
    outputs.append(model_children[i](outputs[-1]))
    
outputs.append(model_children[5](outputs[-1], outputs[3]))
outputs.append(model_children[6](outputs[-1], outputs[2]))
outputs.append(model_children[7](outputs[-1], outputs[1]))
outputs.append(model_children[8](outputs[-1], outputs[0]))
outputs.append(model_children[9](outputs[-1]))

# improved version
for i in range(1, 10):
    outputs.append(model_children[i](outputs[-1]))

outputs.append(model_children[10](outputs[-1], outputs[6]))
outputs.append(model_children[11](outputs[-1], outputs[4]))
outputs.append(model_children[12](outputs[-1], outputs[2]))
outputs.append(model_children[13](outputs[-1], outputs[0]))
outputs.append(model_children[14](outputs[-1]))

for num_block in range(len(outputs)):
    num_block = 9
    x = outputs[num_block].squeeze(0)
    y = torch.sum(x,0)

    plt.imshow(y.detach().cpu().numpy(), cmap='viridis')#gray
    plt.axis('off')
    plt.savefig(f"conv_layer/improved-4/{num_block + 1}.png", pad_inches = 0.1, bbox_inches='tight', dpi=300);

from torch_geometric.nn import SAGEConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv = SAGEConv(hidden_channels, 5)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x
    

model = GCN(1024)
out = model(outputs[9], edge_index)


probs = F.softmax(out_up, dim=1)[0] 
probs = probs.data.numpy().transpose((1, 2, 0))#.detach.cpu().numpy()
mask = np.argmax(probs, axis=2)
plt.imshow(mask)

probs = probs.argmax(dim=1)
probs = pred.data.numpy()
probs = np.reshape(probs, (38, 75))
plt.imshow(probs, cmap=color)
plt.axis('off')
plt.savefig(f"test-120-98.png", bbox_inches='tight', pad_inches = 0, dpi=300)

batch_size, channels, height, width = outputs[8].shape
out_test = outputs[9].view(batch_size, channels, height, width) 
out_up = torch.nn.Upsample(size=(600, 1200), mode='bilinear')(out_test)



'''
Save conv layer variable to file
'''

import pickle

# with open('basic', 'wb') as f:
#     pickle.dump(outputs,f)    
 
with open('conv_layer/basic', 'rb') as f:
    feature_map = (pickle.load(f)) 

with open('conv_layer/improved', 'rb') as f:
    ifeature_map = (pickle.load(f)) 
