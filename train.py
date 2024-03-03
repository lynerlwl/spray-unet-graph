import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop, lr_scheduler
import numpy as np
from tqdm import tqdm

from utility.data_loading import LoadDataset
from model.unet import SUNet
from utility.early_stop import EarlyStopper
from utility.dice_score import dice_loss
from evaluate import evaluate


logging.basicConfig(filename='info.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
dir_img = Path('train_images/img/')
dir_mask = Path('train_images/msk/')
dir_checkpoint = Path('checkpoint/')
load = False
val_percent = 0.1
batch_size = 1
num_epochs = 100
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_checkpoint = True

if __name__ == '__main__':

    # 1. Create dataset
    images_dir = dir_img
    masks_dir = dir_mask
    dataset = LoadDataset(images_dir, masks_dir)

    # Split into train / validation partitions
    n_val = int(np.floor(len(dataset) * val_percent))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False,  **loader_args)

    # 2. Initialise model

    model = SUNet(n_channels=3, n_classes=5, bilinear=False)

    if load != False:
        pretrained_dict = torch.load(load, map_location=device)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        logging.info(f'Model loaded from {load}')

    model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-7, momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # adjust learning rate to maximize Dice score
    early_stopper = EarlyStopper(patience=3, min_delta=10)
    
    # 3. Train model
    
    for epoch in range(num_epochs):
        model.train()
        epochs_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
            for batch in train_loader:
                image = batch['image']
                label = batch['mask']
                
                assert image.shape[1] == model.n_channels, f"Network is defined with {model.n_channels} but images loaded have {image.shape[1]} channels."
                
                image = image.to(device=device, dtype=torch.float)
                label = label.to(device=device, dtype=torch.long)
                
                # forward pass
                prediction = model(image)
                loss = criterion(prediction, label) \
                       + dice_loss(F.softmax(prediction, dim=1).float(),             
                                   F.one_hot(label, model.n_classes).permute(0, 3, 1, 2).float(),
                                   multiclass=True)
                
                # backward pass
                optimizer.zero_grad(set_to_none=True) # Zero out the gradients accumulate during each backward pass
                loss.backward() # Calculate gradients
                optimizer.step() # Update the model's parameters using the computed gradients
                
                pbar.update(image.shape[0])
                epochs_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # evaluation
                val_score = evaluate(model, val_loader, device)
                scheduler.step(val_score)
                
                logging.info(f"Epoch: {epoch+1}, Train loss: {loss.item()}, Validation Dice score: {val_score}")
                
                if early_stopper.early_stop(loss.item()):             
                    break
                
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / f'epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved!')
                
            
