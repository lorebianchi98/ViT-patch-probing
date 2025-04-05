import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.metrics import classification_report
from src.model import LinearProber
from src.dataset import PascalVOCWrapper

classes = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

def train_and_eval(config, out_dir):
    model_name = config['model_name']
    data_root = f'{config["data_root"]}'
    train_datapath = config['train_datapath']
    val_datapath = config['val_datapath']
    
    resize_dim = config['resize_dim']
    lr = config['lr']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    remove_background = config['remove_background']
    
    out_path = os.path.join(out_dir, f'{model_name}.txt')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Setting the random seed...")
    torch.manual_seed(123)

    n_classes = len(classes)
    if not remove_background:
        n_classes += 1
        
    print("Loading the model...")
    model = LinearProber(model_name, n_classes, resize_dim=resize_dim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    

    print("Loading training set...")
    train_voc_dataset = PascalVOCWrapper(data_root, train_datapath, model.image_transforms, model.resize_dim, model.patch_dim)

    print("Loading validation set")
    val_voc_dataset = PascalVOCWrapper(data_root, val_datapath, model.image_transforms, model.resize_dim, model.patch_dim)
    
    print("Creating the DataLoaders...")
    train_dataloader = DataLoader(
        train_voc_dataset,
        batch_size=batch_size,
        shuffle=True,         
        pin_memory=True       
    )

    val_dataloader = DataLoader(
        val_voc_dataset,
        batch_size=batch_size,
        shuffle=False,        
        pin_memory=True      
    )
    
    print("Starting the training loop...")
    len_dataset = len(train_dataloader)
    for n_epoch, epoch in enumerate(range(num_epochs)):
        train_loss = []
        for idx, batch in tqdm(enumerate(train_dataloader), total=len_dataset):
            if idx == len_dataset:
                break
            imgs = batch['img'].to(device)
            grid_maps = batch['grid_segmentation_map'].to(device)
            metadata = batch['metadata']
            
            # flattening the grid segmentation map
            # n_patches x batch_size is used as batch size for the cross-entropy loss
            patches_gt = grid_maps.view(-1)
            
            # calculating prediction for each patch of each image
            logits = model(imgs)
            # flattening the preds to match targets shape
            logits = logits.view(patches_gt.shape[0], n_classes)
            
            if remove_background:
                # we do not calculate gradient on background patches and we re-align labels to match layer predictions
                nonzero_idx = patches_gt != 0
                patches_gt = patches_gt[nonzero_idx] - 1
                logits = logits[nonzero_idx]

            # compute the loss
            loss = criterion(logits, patches_gt)
            train_loss.append(loss.item())
            
            # backward pass
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()        # compute gradients
            optimizer.step()       # update parameters
        
        print(f"Epoch {n_epoch} - Training loss: {sum(train_loss) / len(train_loss)}")
    
    print("Starting the evaluation loop...")
    y_preds = []
    y_true = []

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            imgs = batch['img'].to(device)
            grid_maps = batch['grid_segmentation_map'].to(device)
            metadata = batch['metadata']
            
            # flattening the grid segmentation map
            # n_patches x batch_size is used as batch size for the cross-entropy loss
            patches_gt = grid_maps.view(-1)
            
            # calculating prediction for each patch of each image
            logits = model(imgs)
            # flattening the preds to match targets shape
            logits = logits.view(patches_gt.shape[0], n_classes)
            
            if remove_background:
                # we do not calculate gradient on background patches and we re-align labels to match layer predictions
                nonzero_idx = patches_gt != 0
                patches_gt = patches_gt[nonzero_idx] - 1
                logits = logits[nonzero_idx]

            preds = logits.argmax(dim=-1)
            
            # storing gt and predictions
            y_true += patches_gt.tolist()
            y_preds += preds.tolist()
            
    report = classification_report(y_true, y_preds, target_names=classes)
    print("Classification Report:\n", report)

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as file:
        file.write(report)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Training configuration')
    parser.add_argument('--out_dir', type=str, default="results", help='Out directory where the results will be stored')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    train_and_eval(config, args.out_dir)
    
    
if __name__ == '__main__':
    main()