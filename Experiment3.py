from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from datetime import datetime
from Model3 import Net, get_optimizer_and_scheduler


# Train Phase transformations
train_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                       transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])

# Test Phase transformations
test_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])

"""# Dataset and Creating Train/Test Split"""

train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

"""# Dataloader Arguments & Test/Train Dataloaders

"""

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

"""# The model
Let's start with the model we first saw
"""
dropout_value = 0.1



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net().to(device)
summary(model, input_size=(1, 28, 28))

"""# Training and Testing

Looking at logs can be boring, so we'll introduce **tqdm** progressbar to get cooler logs.

Let's write train and test functions
"""

from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []


def train(model, device, train_loader, optimizer, scheduler, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  running_loss = 0.0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    # train_losses.append(loss.item())

    # Backpropagation
    loss.backward()
    optimizer.step()

    scheduler.step()
    running_loss += loss.item()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(
        desc=f'Epoch {epoch} Loss={loss.item():.4f} '
             f'Batch_id={batch_idx} '
             f'Accuracy={100*correct/processed:0.2f}'
    )

    # train_losses.append(loss / len(train_loader))
    train_losses.append(loss.item()) 
    train_acc.append(100. * correct / processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    test_losses.append(test_loss)
    test_acc.append(accuracy)

    print(f"\nTest set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({accuracy:.2f}%)\n")

    return test_loss   



model =  Net().to(device)
optimizer, scheduler = get_optimizer_and_scheduler(model,len(train_loader), EPOCHS=15)

if __name__ == "__main__":
    EPOCHS = 15
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch+1)
        print(datetime.now().strftime("%d%m%Y-%H%M"))
        train(model, device, train_loader, optimizer, scheduler, epoch+1)
        print(datetime.now().strftime("%d%m%Y-%H%M"))
        val_loss=test(model, device, test_loader)
        
        print("-----------------------------------------------")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,2,figsize=(15,10))

    # axs[0, 0].plot(train_losses)
    axs[0, 0].plot([loss for loss in train_losses], label='Training Loss')
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].legend()

    axs[1, 0].plot([acc for acc in train_acc], label="Train Accuracy")
    axs[1, 0].set_title("Training Accuracy")
    axs[1, 0].legend()
    
    axs[0, 1].plot([loss for loss in test_losses], label="Test Loss", color="orange")
    axs[0, 1].set_title("Test Loss")
    axs[0, 1].legend()

    axs[1, 1].plot([acc for acc in test_acc], label="Test Accuracy", color="green")
    axs[1, 1].set_title("Test Accuracy")
    axs[1, 1].legend()
        
    # Format timestamp as DDMMYYYY-HHMM
    
    timestamp = datetime.now().strftime("%d%m%Y-%H%M")

    # Save with timestamp in filename
    filename = f"training_results-{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    print(f"Plot saved as {filename}")
    # plt.show()
