import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_optimizer_scheduler(model, learning_rate=5e-5, total_steps=10000, warmup_steps=1000):
    """
    Create optimizer and learning rate scheduler
    
    Args:
        model (nn.Module): Model to optimize
        learning_rate (float): Base learning rate
        total_steps (int): Total training steps
        warmup_steps (int): Number of warmup steps
    
    Returns:
        tuple: Optimizer and Learning Rate Scheduler
    """
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01
    )
    
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-6
    )
    
    # Combined scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    return optimizer, scheduler

def train_epoch(
    model, 
    dataloader, 
    optimizer, 
    scheduler, 
    criterion, 
    device
):
    """
    Train the model for one epoch
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        criterion (nn.Module): Loss function
        device (torch.device): Training device
    
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Prepare batch
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimize
        optimizer.step()
        scheduler.step()
        
        # Track loss
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(
    model, 
    dataloader, 
    criterion, 
    device
):
    """
    Evaluate model performance
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Validation/Test data
        criterion (nn.Module): Loss function
        device (torch.device): Evaluation device
    
    Returns:
        tuple: Average loss and accuracy
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Prepare batch
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Track loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy

def train_model(
    model, 
    train_dataset, 
    test_dataset, 
    num_epochs=5, 
    batch_size=32, 
    learning_rate=5e-5
):
    """
    Complete model training pipeline
    
    Args:
        model (nn.Module): Model to train
        train_dataset (Dataset): Training dataset
        test_dataset (Dataset): Test dataset
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Base learning rate
    
    Returns:
        dict: Training history with losses and accuracies
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Total training steps
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_scheduler(
        model, 
        learning_rate=learning_rate,
        total_steps=total_steps,
        warmup_steps=warmup_steps
    )
    
    # Training history tracking
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Train epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, 
            criterion, device
        )
        history['train_loss'].append(train_loss)
        
        # Evaluate
        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, device
        )
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("-" * 40)
    
    # Plot training history
    plt.figure(figsize=(12,4))
    
    # Training Loss
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Test Accuracy
    plt.subplot(1,2,2)
    plt.plot(history['test_accuracy'], label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    return history

def save_model(model, path='children_book_age_classifier.pth'):
    """
    Save trained model
    
    Args:
        model (nn.Module): Trained model
        path (str): Save path
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path='children_book_age_classifier.pth'):
    """
    Load trained model weights
    
    Args:
        model (nn.Module): Model architecture
        path (str): Path to saved weights
    
    Returns:
        nn.Module: Model with loaded weights
    """
    model.load_state_dict(torch.load(path))
    return model
