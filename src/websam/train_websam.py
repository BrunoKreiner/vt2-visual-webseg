import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from segment_anything import build_sam_vit_b
import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

BATCH_SIZE = 2
grad_accum_steps = 1

# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()

def create_run_directory(base_path="../../models/websam", name=None):
    """Create a new directory for this training run"""
    os.makedirs(base_path, exist_ok=True)

    if name:
        run_dir = os.path.join(base_path, name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_path, f"run_{timestamp}")
    
    os.makedirs(run_dir, exist_ok=True)

    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return run_dir, models_dir

def plot_metrics(metrics, run_dir, epoch):
    """Plot training and validation metrics, ensuring x and y axes are aligned"""
    metric_keys = ['loss', 'pixel_accuracy', 'f1', 'pb3', 'rb3', 'fb3']
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    
    # Use the actual metrics data to determine epochs
    train_data_len = len(metrics['train']['loss'])
    val_data_len = len(metrics['val']['loss'])
    
    # Make sure we have equal length arrays for plotting
    train_epochs = list(range(1, train_data_len + 1))
    val_epochs = list(range(1, val_data_len + 1))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metric_keys):
        if metric in metrics['train']:
            axes[i].plot(train_epochs, metrics['train'][metric], 'b-', label=f'Training')
        if metric in metrics['val']:
            axes[i].plot(val_epochs, metrics['val'][metric], 'r-', label=f'Validation')
        
        axes[i].set_title(f'{metric.capitalize()}')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(metric)
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'plots', f'metrics_epoch_{epoch}.png'))
    plt.close()

def visualize_predictions(images, targets, predictions, epoch, run_dir, num_samples=3):
    """Visualize a few sample predictions"""
    os.makedirs(os.path.join(run_dir, 'visualizations'), exist_ok=True)
    
    for i in range(min(num_samples, len(images))):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        img = images[i].cpu().permute(1, 2, 0).numpy()
        if img.max() > 1.0:
            img = img / 255.0
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # Ground truth - ensure it's a proper 2D tensor for visualization
        target = targets[i]
        # If we have a multi-segment mask (3D tensor), we need to reduce it to 2D
        if len(target.shape) == 3:
            target = target[0]
        
        # Further ensure it's a 2D tensor (no extra dimensions)
        if len(target.shape) > 2:
            target = target.squeeze()
            
        axes[1].imshow(target.cpu().numpy(), cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        pred = predictions[i]
        # Handle the case where prediction is also multi-segment
        if len(pred.shape) == 3:
            # Take the first prediction or combine all predictions
            pred = pred[0]
            
        # Further ensure it's a 2D tensor (no extra dimensions)
        if len(pred.shape) > 2:
            pred = pred.squeeze()
            
        pred_display = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()
        axes[2].imshow(pred_display, cmap='gray')
        axes[2].set_title("Prediction")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'visualizations', f'epoch_{epoch}_sample_{i}.png'))
        plt.close()

def plot_training_progress(train_metrics_history, val_metrics_history, run_dir):
    """Plot training and validation metrics over epochs, ensuring x and y axes are aligned"""
    train_epochs = range(1, len(train_metrics_history['loss']) + 1)
    val_epochs = range(1, len(val_metrics_history['loss']) + 1)
    
    metrics = ['loss', 'pixel_accuracy', 'f1', 'pb3', 'rb3', 'fb3']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        axes[i].plot(train_epochs, train_metrics_history[metric], 'b-', label=f'Training {metric}')
        axes[i].plot(val_epochs, val_metrics_history[metric], 'r-', label=f'Validation {metric}')
        axes[i].set_title(f'{metric.capitalize()}')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(metric)
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_progress.png'))
    plt.close()


# Define the combined BCE and IoU Loss
class BCEIoULoss(nn.Module):
    def __init__(self):
        super(BCEIoULoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)

        # IoU loss
        inputs_sigmoid = torch.sigmoid(inputs)
        #print(f"inputs_sigmoid shape: {inputs_sigmoid.shape}, targets shape: {targets.shape}")
        intersection = (inputs_sigmoid * targets).sum((1, 2))
        total = (inputs_sigmoid + targets).sum((1, 2))
        union = total - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1 - iou.mean()
        total_loss = bce_loss + iou_loss.mean()
        #print(f"BCE: {bce_loss.item()}, IoU: {iou_loss.item()}, Total: {total_loss.item()}")

        return total_loss

class WebSegDataset(Dataset):
    def __init__(self, root_dir, image_size=1024, max_samples=None):  # Reduced size for efficiency
        self.image_dir = os.path.join(root_dir, "images")
        self.box_dir = os.path.join(root_dir, "boxes")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.image_size = image_size
        self.image_filenames = sorted(os.listdir(self.image_dir))  

        if max_samples is not None:
            self.image_filenames = self.image_filenames[:max_samples]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        box_path = os.path.join(self.box_dir, self.image_filenames[idx].replace(".png", ".npy"))
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx].replace(".png", ".npy"))

        # Load image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # Normalize
        
        # Load bounding box
        #boxes = np.load(box_path)
        
        # Load mask
        mask = np.load(mask_path)
        
        # Debug: Print mask shape
        #print(f"Loaded mask shape: {mask.shape}, type: {type(mask)}, min: {mask.min()}, max: {mask.max()}")

        # Convert to tensors efficiently
        image = torch.tensor(image).permute(2, 0, 1).contiguous()  # Contiguous for speed
        #boxes = torch.tensor(boxes, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        # Debug: Print tensor mask shape
        #print(f"Tensor mask shape: {mask.shape}")

        return image, mask #image, boxes, mask
    
def collate_fn(batch):
    #images, boxes, masks = zip(*batch)
    images, masks = zip(*batch)
    images = torch.stack(images)
    return images, masks #images, boxes, masks

def calculate_metrics(pred_masks, target_masks):
    """Calculate PB3, RB3, F*B3, F1, and Pixel Accuracy"""
    #print("calculate_metrics")
    # Convert logits to binary predictions
    pred_masks = (torch.sigmoid(pred_masks) > 0.5)  # Returns boolean tensor
    target_masks = (target_masks > 0.5)  # Convert target to boolean tensor

    #print(pred_masks.shape, target_masks.shape)
    
    # Pixel Accuracy
    pixel_accuracy = (pred_masks == target_masks).float().mean()
    
    #print(pixel_accuracy)

    # Convert to float for calculations
    pred_float = pred_masks.float()
    target_float = target_masks.float()
    
    # F1 Score
    intersection = (pred_float * target_float).sum()
    precision = intersection / (pred_float.sum() + 1e-6)
    recall = intersection / (target_float.sum() + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # B3 metrics
    pb3 = precision  # For pixels, PB3 is equivalent to precision
    rb3 = recall    # For pixels, RB3 is equivalent to recall
    fb3 = f1        # For pixels, F*B3 is equivalent to F1
    
    return {
        'pixel_accuracy': pixel_accuracy.item(),
        'f1': f1.item(),
        'pb3': pb3.item(),
        'rb3': rb3.item(),
        'fb3': fb3.item()
    }
class MetricTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'loss': [],
            'pixel_accuracy': [],
            'f1': [],
            'pb3': [],
            'rb3': [],
            'fb3': []
        }
    
    def update(self, metrics_dict):
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
    
    def get_averages(self):
        return {key: np.mean(values) for key, values in self.metrics.items()}

def load_metrics_up_to_epoch(metrics_file, max_epoch):
    """Load metrics from CSV but only up to specified epoch"""
    epoch_metrics = {
        'train': {'loss': [], 'pixel_accuracy': [], 'f1': [], 'pb3': [], 'rb3': [], 'fb3': []},
        'val': {'loss': [], 'pixel_accuracy': [], 'f1': [], 'pb3': [], 'rb3': [], 'fb3': []}
    }
    
    if not os.path.exists(metrics_file):
        return epoch_metrics
    
    df = pd.read_csv(metrics_file)
    if df.empty:
        return epoch_metrics
    
    # Filter to only include epochs up to max_epoch
    df = df[df['epoch'] <= max_epoch]
    
    # Process train metrics
    train_df = df[df['mode'] == 'train']
    for metric in epoch_metrics['train'].keys():
        col_name = f"train_{metric}"
        if col_name in train_df.columns:
            epoch_metrics['train'][metric] = train_df[col_name].tolist()
    
    # Process validation metrics
    val_df = df[df['mode'] == 'val']
    for metric in epoch_metrics['val'].keys():
        col_name = f"val_{metric}"
        if col_name in val_df.columns:
            epoch_metrics['val'][metric] = val_df[col_name].tolist()
    
    return epoch_metrics

def resume_training(run_dir, model, optimizer):
    """Load checkpoint and determine the proper epoch to resume from"""
    models_dir = os.path.join(run_dir, "models")
    metrics_file = os.path.join(run_dir, 'metrics.csv')
    
    # Find the checkpoint to load
    checkpoint_path = None
    checkpoints = [f for f in os.listdir(models_dir) if f.startswith('checkpoint_epoch_')]
    
    if checkpoints:
        # Parse the epoch number from filename - THIS IS ONE-INDEXED
        checkpoint_epoch_filename = max([int(f.split('_')[-1].split('.')[0]) for f in checkpoints])
        checkpoint_path = os.path.join(models_dir, f'checkpoint_epoch_{checkpoint_epoch_filename}.pth')
    
    if not checkpoint_path and os.path.exists(os.path.join(models_dir, 'best_model.pth')):
        checkpoint_path = os.path.join(models_dir, 'best_model.pth')
    
    if not checkpoint_path:
        raise FileNotFoundError("No checkpoints found to resume from")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get the checkpoint epoch and best loss - THIS IS ZERO-INDEXED
    checkpoint_epoch = checkpoint['epoch']
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    best_loss = checkpoint['best_loss']
    
    # The checkpoint epoch (zero-indexed) = the one-indexed epoch that was completed
    completed_epoch = checkpoint_epoch
    # The next epoch to train (one-indexed) is the completed epoch + 1
    next_epoch_one_indexed = completed_epoch + 1
    
    print(f"Checkpoint is from completed epoch {completed_epoch} (internal index: {checkpoint_epoch})")
    print(f"Resuming with epoch {next_epoch_one_indexed} (internal index: {checkpoint_epoch})")
    
    # For cleanup, we need one-indexed values to match filenames
    cleanup_files_for_resuming(run_dir, next_epoch_one_indexed)
    
    # Load metrics up to the completed epoch
    epoch_metrics = load_metrics_up_to_epoch(metrics_file, completed_epoch)
    
    # For the scheduler, use the zero-indexed value
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, last_epoch=checkpoint_epoch)
    
    # Return the zero-indexed start_epoch for internal use
    return completed_epoch, best_loss, epoch_metrics, scheduler

def validate(model, val_loader, criterion, device, epoch=0, run_dir=None, visualize=False):
    model.eval()
    val_tracker = MetricTracker()
    
    # For visualization
    visualization_done = False
    
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f"Validating") as pbar:
            for step, (images, masks) in enumerate(val_loader):
                images = images.to(device)
                masks = [m.to(device) for m in masks]
                
                batched_input = [
                    {
                        "image": image,
                        "boxes": None,
                        "original_size": (image.shape[-2], image.shape[-1])
                    } for image in images
                ]
                outputs = model(batched_input, multimask_output=False)
                
                # Visualize the first batch if enabled
                if visualize and run_dir is not None and not visualization_done:
                    sample_images = images[:min(3, len(images))]
                    sample_targets = []
                    sample_preds = []
                    
                    for i in range(min(3, len(outputs))):
                        # Get first prediction (same as in training)
                        pred_mask = outputs[i]["low_res_logits"][0]  # Shape [1, 256, 256]
                        
                        # Create combined ground truth mask (same as in training)
                        target_mask = masks[i]
                        # If the mask is already 2D (single mask), use it directly
                        if len(target_mask.shape) == 2:
                            combined_gt = (target_mask > 0.5).float()
                        # If it's a 3D tensor (multiple masks), combine them
                        elif len(target_mask.shape) == 3:
                            combined_gt = torch.any(target_mask > 0.5, dim=0).float()
                        else:
                            print(f"Unexpected mask shape: {target_mask.shape}")
                            # Handle unexpected shape - create an empty mask
                            combined_gt = torch.zeros((1024, 1024), device=target_mask.device)
                        
                        # Resize the combined target to match pred size
                        h, w = pred_mask.shape[-2:]
                        target_mask_resized = F.interpolate(
                            combined_gt.unsqueeze(0).unsqueeze(0),
                            size=(h, w),
                            mode='bilinear'
                        ).squeeze(0)  # Shape [1, h, w]
                        
                        sample_targets.append(target_mask_resized.squeeze(0))  # Convert to [h, w] for visualization
                        sample_preds.append(pred_mask.squeeze(0))  # Convert to [h, w] for visualization
                    
                    visualize_predictions(sample_images, sample_targets, sample_preds, epoch, run_dir)
                    visualization_done = True
                
                # Calculate metrics
                batch_metrics = []
                for output, target_mask in zip(outputs, masks):
                    # Create combined ground truth mask
                    # If the mask is already 2D (single mask), use it directly
                    if len(target_mask.shape) == 2:
                        combined_gt = (target_mask > 0.5).float()
                    # If it's a 3D tensor (multiple masks), combine them
                    elif len(target_mask.shape) == 3:
                        combined_gt = torch.any(target_mask > 0.5, dim=0).float()
                    else:
                        print(f"Unexpected mask shape: {target_mask.shape}")
                        # Handle unexpected shape - create an empty mask
                        combined_gt = torch.zeros((1024, 1024), device=target_mask.device)
                    
                    # Resize to match prediction resolution (256x256)
                    h, w = output["low_res_logits"][0].shape[-2:]
                    target_mask_resized = F.interpolate(
                        combined_gt.unsqueeze(0).unsqueeze(0), 
                        size=(h, w), 
                        mode='bilinear'
                    ).squeeze(0)  # Keep channel dim
                    
                    # Take first prediction
                    pred_mask = output["low_res_logits"][0]  # Shape [1, 256, 256]
                    
                    loss = criterion(pred_mask, target_mask_resized)
                    metrics = calculate_metrics(pred_mask, target_mask_resized)
                    metrics['loss'] = loss.item()
                    batch_metrics.append(metrics)
                
                # Average batch metrics
                avg_batch_metrics = {k: np.mean([m[k] for m in batch_metrics]) for k in batch_metrics[0].keys()}
                val_tracker.update(avg_batch_metrics)
                
                # Update progress bar with metrics
                metrics_str = ", ".join([f"{k}: {avg_batch_metrics[k]:.4f}" for k in sorted(avg_batch_metrics.keys())])
                pbar.set_postfix_str(metrics_str)
                pbar.update(1)
                
                # Clean up memory
                del images, masks, outputs
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
    
    return val_tracker.get_averages()

def save_checkpoint(model, optimizer, epoch, best_loss, metrics, run_dir, scheduler, is_best=False):
    """Save model checkpoint"""
    models_dir = os.path.join(run_dir, "models")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'metrics': metrics
    }
    
    if is_best:
        filename = os.path.join(models_dir, 'best_model.pth')
    else:
        filename = os.path.join(models_dir, f'checkpoint_epoch_{epoch+1}.pth')
    
    torch.save(checkpoint, filename)

def cleanup_files_for_resuming(run_dir, start_epoch_one_indexed):
    """Remove visualization and plot files for epochs that will be overwritten.
    start_epoch_one_indexed is the one-indexed epoch number to start cleaning from."""
    
    # Clean up visualization files
    vis_dir = os.path.join(run_dir, 'visualizations')
    if os.path.exists(vis_dir):
        for file in os.listdir(vis_dir):
            # Parse epoch number from filenames like 'epoch_11_sample_0.png'
            if file.startswith('epoch_'):
                try:
                    epoch_num = int(file.split('_')[1])
                    if epoch_num >= start_epoch_one_indexed:
                        os.remove(os.path.join(vis_dir, file))
                        print(f"Removed visualization file: {file}")
                except (IndexError, ValueError):
                    continue
    
    # Clean up plot files
    plots_dir = os.path.join(run_dir, 'plots')
    if os.path.exists(plots_dir):
        for file in os.listdir(plots_dir):
            # Parse epoch number from filenames like 'metrics_epoch_11.png'
            if file.startswith('metrics_epoch_'):
                try:
                    epoch_num = int(file.split('_')[-1].split('.')[0])
                    if epoch_num >= start_epoch_one_indexed:
                        os.remove(os.path.join(plots_dir, file))
                        print(f"Removed plot file: {file}")
                except (IndexError, ValueError):
                    continue
    
    # Update metrics.csv to remove epochs that will be overwritten
    metrics_file = os.path.join(run_dir, 'metrics.csv')
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        # Keep only rows with epoch < start_epoch_one_indexed
        df = df[df['epoch'] < start_epoch_one_indexed]
        df.to_csv(metrics_file, index=False)
        print(f"Updated metrics.csv to include only epochs up to {start_epoch_one_indexed-1}")

def update_metrics_csv(metrics_dict, epoch, run_dir, mode='train'):
    """Update the metrics CSV file with new epoch results"""
    metrics_file = os.path.join(run_dir, 'metrics.csv')
    
    # Prepare the metrics row
    row_dict = {
        'epoch': epoch,
        'mode': mode
    }
    row_dict.update({f"{mode}_{k}": v for k, v in metrics_dict.items()})
    
    # If file exists, append to it
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        # Remove any existing row for this epoch and mode to avoid duplicates
        df = df[(df['epoch'] != epoch) | (df['mode'] != mode)]
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([row_dict])
    
    # Save to CSV
    df.to_csv(metrics_file, index=False)

# Modified training loop
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, 
          run_dir=None, start_epoch=0, best_loss=float('inf'), epoch_metrics=None):
    # Create run directory if not provided
    if run_dir is None:
        run_dir, models_dir = create_run_directory(base_path="../../models/websam", name=None)
        print(f"Saving results to: {run_dir}")
    else:
        models_dir = os.path.join(run_dir, "models")
    
    # Initialize metrics tracking if not provided
    if epoch_metrics is None:
        epoch_metrics = {
            'train': {'loss': [], 'pixel_accuracy': [], 'f1': [], 'pb3': [], 'rb3': [], 'fb3': []},
            'val': {'loss': [], 'pixel_accuracy': [], 'f1': [], 'pb3': [], 'rb3': [], 'fb3': []}
        }
    print("start_epoch:", start_epoch)
    print(range(start_epoch, num_epochs))
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_tracker = MetricTracker()
        
        with tqdm(total=len(train_loader) * BATCH_SIZE, desc=f"Epoch {epoch+1}/{num_epochs} - Training") as pbar:
            for step, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks = [m.to(device) for m in masks]
                
                # Use mixed precision for faster training
                with torch.cuda.amp.autocast():
                    batched_input = [
                        {
                            "image": image,
                            "boxes": None,
                            "original_size": (image.shape[-2], image.shape[-1])
                        } for image in images
                    ]
                    outputs = model(batched_input, multimask_output=False)
                    
                    batch_metrics = []
                    losses = []
                    for output, target_mask in zip(outputs, masks):
                        # Create combined ground truth mask
                        # If the mask is already 2D (single mask), use it directly
                        if len(target_mask.shape) == 2:
                            combined_gt = (target_mask > 0.5).float()
                        # If it's a 3D tensor (multiple masks), combine them
                        elif len(target_mask.shape) == 3:
                            combined_gt = torch.any(target_mask > 0.5, dim=0).float()
                        else:
                            print(f"Unexpected mask shape: {target_mask.shape}")
                            # Handle unexpected shape - create an empty mask
                            combined_gt = torch.zeros((1024, 1024), device=target_mask.device)
                        
                        # Resize to match prediction resolution (256x256)
                        h, w = output["low_res_logits"][0].shape[-2:]
                        target_mask_resized = F.interpolate(
                            combined_gt.unsqueeze(0).unsqueeze(0), 
                            size=(h, w), 
                            mode='bilinear'
                        ).squeeze(0)  # Keep channel dim
                        
                        # Take first prediction and keep channel dimension
                        pred_mask = output["low_res_logits"][0]  # Shape should be [1, 256, 256]
                        loss = criterion(pred_mask, target_mask_resized)
                        losses.append(loss)
                        
                        metrics = calculate_metrics(pred_mask, target_mask_resized)
                        metrics['loss'] = loss.item()
                        batch_metrics.append(metrics)
                    
                    loss = torch.stack(losses).mean() / grad_accum_steps

                # Gradient accumulation for larger effective batch sizes
                scaler.scale(loss).backward()
                
                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # Update metrics and logging
                avg_batch_metrics = {k: np.mean([m[k] for m in batch_metrics]) for k in batch_metrics[0].keys()}
                train_tracker.update(avg_batch_metrics)
                metrics_str = ", ".join([f"{k}: {avg_batch_metrics[k]:.4f}" for k in sorted(avg_batch_metrics.keys())])
                pbar.set_postfix_str(metrics_str)
                pbar.update(BATCH_SIZE)

                # Clean up memory
                del images, masks, outputs, loss
                if step % 10 == 0:  # Periodically clean up memory more aggressively
                    torch.cuda.empty_cache()
                    gc.collect()

        train_metrics = train_tracker.get_averages()
        for k, v in train_metrics.items():
            if k in epoch_metrics['train']:
                epoch_metrics['train'][k].append(v)
        
        scheduler.step()
        
        # Validation phase
        val_metrics = validate(model, val_loader, criterion, device, epoch=epoch, 
                       run_dir=run_dir, visualize=(epoch % 1 == 0))
        for k, v in val_metrics.items():
            if k in epoch_metrics['val']:
                epoch_metrics['val'][k].append(v)

        update_metrics_csv(train_metrics, epoch, run_dir, mode='train')
        update_metrics_csv(val_metrics, epoch, run_dir, mode='val')

        # Save checkpoint if best model
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            save_checkpoint(model, optimizer, epoch, best_loss, val_metrics, run_dir, scheduler, is_best=True)
        
        # Save regular checkpoint every 1 epochs
        if (epoch + 1) % 1 == 0:
            save_checkpoint(model, optimizer, epoch, best_loss, val_metrics, run_dir, scheduler, is_best=False)
        
        plot_metrics(epoch_metrics, run_dir, epoch)
        plot_training_progress(epoch_metrics['train'], epoch_metrics['val'], run_dir)

        # Print epoch summary
        print(f"\nEpoch [{epoch}/{num_epochs}] Summary:")
        print("Training metrics:", {k: f"{v:.4f}" for k, v in train_metrics.items()})
        print("Validation metrics:", {k: f"{v:.4f}" for k, v in val_metrics.items()})

# Add this main block to handle resuming
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train or resume WebSAM training')
    parser.add_argument('--resume', type=str, default=None, help='Path to run directory to resume training from')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--imgsz', type=int, default=1024, help='Image size for training')
    parser.add_argument('--data', type=str, default="/home/bruno/vt2-visual-webseg/data/webis-webseg-20-sam-big-segments-full-better", 
                        help='Path to dataset directory')
    parser.add_argument('--project', type=str, default="../../models/websam", 
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default=None, 
                        help='Run name (default: timestamp)')
    parser.add_argument('--checkpoint', type=str, default="./models/websam/official_checkpoints/sam_vit_b_01ec64.pth", 
                        help='Path to SAM checkpoint')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of samples to use for training and validation (for quick testing)')
    args = parser.parse_args()
    
    # Update globals if specified in args
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.grad_accum_steps:
        grad_accum_steps = args.grad_accum_steps
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # Initialize model
    sam = build_sam_vit_b(
        checkpoint=args.checkpoint,
        strict_weights=False,
        freeze_encoder=args.freeze_encoder
    )
    
    if torch.cuda.is_available():
        print("Using CUDA")
        sam = sam.cuda()
    
    # Setup datasets and dataloaders
    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "val")
    
    train_dataset = WebSegDataset(root_dir=train_dir, image_size=args.imgsz, max_samples=args.max_samples)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dataset = WebSegDataset(root_dir=val_dir, image_size=args.imgsz, max_samples=args.max_samples)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Initialize optimizer, scheduler, and criterion
    optimizer = optim.AdamW(sam.parameters(), lr=args.lr)
    num_epochs = args.epochs
    criterion = BCEIoULoss()
    
    # Create custom run directory if name is provided
    if args.name:
        base_path = args.project
        os.makedirs(base_path, exist_ok=True)
        run_dir = os.path.join(base_path, args.name)
        os.makedirs(run_dir, exist_ok=True)
        models_dir = os.path.join(run_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
    else:
        run_dir = None
    
    # Resume training if specified
    if args.resume:
        run_dir = args.resume
        last_epoch, best_loss, epoch_metrics, scheduler = resume_training(run_dir, sam, optimizer)
        print(f"Resuming training from epoch {last_epoch + 1}")
        print(f"best_loss: {best_loss}")
        
        # Resume training
        train(sam, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device,
            run_dir=run_dir, start_epoch=last_epoch + 1, best_loss=best_loss, epoch_metrics=epoch_metrics)
    else:
        # Start training from scratch
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        if args.name:
            train(sam, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, run_dir=run_dir)
        else:
            train(sam, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)