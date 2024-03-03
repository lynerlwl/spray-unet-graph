from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from matplotlib import pyplot as plt

'''
feature map
'''

for i in range(0, 5, 1):
    img = io.imread(f'conv_layer/visualisation/basic-unet/{i}.png', as_gray=True)
    img_entropy = entropy(img, disk(3))
    # plt.hist(img_entropy.flat)
    binary = img_entropy >= 0.01
    plt.imshow(binary)
    plt.axis('off')
    plt.savefig(f"conv_layer/visualisation/basic-unet/entropy/{i}.png", bbox_inches='tight', pad_inches = 0, dpi=300)
    
for i in range(1, 10, 2):
    img = io.imread(f'conv_layer/improved-4/gray/{i}.png', as_gray=True)
    img_entropy = entropy(img, disk(3))
    # plt.hist(img_entropy.flat)
    binary = img_entropy >= 1
    plt.imshow(binary)
    plt.axis('off')
    plt.savefig(f"conv_layer/improved-4/entropy/{i}.png", bbox_inches='tight', pad_inches = 0, dpi=300)
    
    i=10
    img = io.imread(f'conv_layer/improved-4/gray/{i}.png', as_gray=True)
    img_entropy = entropy(img, disk(3))
    binary = img_entropy >= 1
    plt.imshow(binary)
    plt.axis('off')
    plt.savefig(f"conv_layer/improved-4/entropy/{i}.png", bbox_inches='tight', pad_inches = 0, dpi=300)
    
'''
raw image
'''  
img = io.imread('data/spray_ horizontal/f_01213.png', as_gray=True)
img_entropy = entropy(img, disk(3))
# plt.hist(img_entropy.flat)
th = [0.01, 1, 3, 4]
for i in th:
    binary = img_entropy >= i
    plt.imshow(binary)
    plt.axis('off')
    plt.savefig(f"show/entropy_{i}.png", bbox_inches='tight', pad_inches = 0, dpi=300)



