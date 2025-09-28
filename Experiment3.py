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
                                       transforms.RandomRotation((-10.0, 10.0), fill=(1,)),
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


SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args_train = dict(shuffle=True, batch_size=64, num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args_train)

dataloader_args_test = dict(shuffle=False, batch_size=1000, num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
# test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args_test)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net().to(device)
summary(model, input_size=(1, 28, 28))


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

#LOGS

"""

(.venv) PS C:\Raghu\MyLearnings\ERA_V4\S6-21092025\code\mnist_digits_experiments> python Experiment3.py
CUDA Available? True
cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
            Conv2d-3           [-1, 16, 28, 28]           1,168
       BatchNorm2d-4           [-1, 16, 28, 28]              32
         MaxPool2d-5           [-1, 16, 14, 14]               0
            Conv2d-6           [-1, 16, 14, 14]           2,320
       BatchNorm2d-7           [-1, 16, 14, 14]              32
         MaxPool2d-8             [-1, 16, 7, 7]               0
            Conv2d-9             [-1, 28, 7, 7]           4,060
      BatchNorm2d-10             [-1, 28, 7, 7]              56
           Conv2d-11             [-1, 10, 7, 7]             290
AdaptiveAvgPool2d-12             [-1, 10, 1, 1]               0
================================================================
Total params: 8,054
Trainable params: 8,054
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.39
Params size (MB): 0.03
Estimated Total Size (MB): 0.42
----------------------------------------------------------------
EPOCH: 1
28092025-2126
Epoch 1 Loss=0.2574 Batch_id=937 Accuracy=81.91: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:53<00:00, 17.68it/s]
28092025-2127

Test set: Average loss: 0.1338, Accuracy: 9606/10000 (96.06%)

-----------------------------------------------
EPOCH: 2
28092025-2127
Epoch 2 Loss=0.1229 Batch_id=937 Accuracy=95.09: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:59<00:00, 15.79it/s] 
28092025-2128

Test set: Average loss: 0.0793, Accuracy: 9752/10000 (97.52%)

-----------------------------------------------
EPOCH: 3
28092025-2128
Epoch 3 Loss=0.2285 Batch_id=937 Accuracy=96.20: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:02<00:00, 14.91it/s] 
28092025-2129

Test set: Average loss: 0.0607, Accuracy: 9812/10000 (98.12%)

-----------------------------------------------
EPOCH: 4
28092025-2129
Epoch 4 Loss=0.0736 Batch_id=937 Accuracy=96.83: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:07<00:00, 13.82it/s] 
28092025-2130

Test set: Average loss: 0.0666, Accuracy: 9778/10000 (97.78%)

-----------------------------------------------
EPOCH: 5
28092025-2130
Epoch 5 Loss=0.2275 Batch_id=937 Accuracy=96.99: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:07<00:00, 13.86it/s] 
28092025-2132

Test set: Average loss: 0.1050, Accuracy: 9655/10000 (96.55%)

-----------------------------------------------
EPOCH: 6
28092025-2132
Epoch 6 Loss=0.0287 Batch_id=937 Accuracy=97.11: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:09<00:00, 13.53it/s] 
28092025-2133

Test set: Average loss: 0.0636, Accuracy: 9796/10000 (97.96%)

-----------------------------------------------
EPOCH: 7
28092025-2133
Epoch 7 Loss=0.0107 Batch_id=937 Accuracy=97.33: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:09<00:00, 13.44it/s] 
28092025-2134

Test set: Average loss: 0.0555, Accuracy: 9823/10000 (98.23%)

-----------------------------------------------
EPOCH: 8
28092025-2134
Epoch 8 Loss=0.2168 Batch_id=937 Accuracy=97.48: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:06<00:00, 14.05it/s] 
28092025-2135

Test set: Average loss: 0.0471, Accuracy: 9859/10000 (98.59%)

-----------------------------------------------
EPOCH: 9
28092025-2135
Epoch 9 Loss=0.0603 Batch_id=937 Accuracy=97.63: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:07<00:00, 13.88it/s] 
28092025-2136

Test set: Average loss: 0.0292, Accuracy: 9905/10000 (99.05%)

-----------------------------------------------
EPOCH: 10
28092025-2137
Epoch 10 Loss=0.0080 Batch_id=937 Accuracy=97.84: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:10<00:00, 13.23it/s] 
28092025-2138

Test set: Average loss: 0.0399, Accuracy: 9869/10000 (98.69%)

-----------------------------------------------
EPOCH: 11
28092025-2138
Epoch 11 Loss=0.0658 Batch_id=937 Accuracy=98.16: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:07<00:00, 13.88it/s] 
28092025-2139

Test set: Average loss: 0.0293, Accuracy: 9909/10000 (99.09%)

-----------------------------------------------
EPOCH: 12
28092025-2139
Epoch 12 Loss=0.0154 Batch_id=937 Accuracy=98.27: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:09<00:00, 13.47it/s] 
28092025-2140

Test set: Average loss: 0.0243, Accuracy: 9926/10000 (99.26%)

-----------------------------------------------
EPOCH: 13
28092025-2140
Epoch 13 Loss=0.0801 Batch_id=937 Accuracy=98.50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:08<00:00, 13.71it/s] 
28092025-2141

Test set: Average loss: 0.0227, Accuracy: 9926/10000 (99.26%)

-----------------------------------------------
EPOCH: 14
28092025-2142
Epoch 14 Loss=0.0195 Batch_id=937 Accuracy=98.64: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:11<00:00, 13.05it/s] 
28092025-2143

Test set: Average loss: 0.0215, Accuracy: 9931/10000 (99.31%)

-----------------------------------------------
EPOCH: 15
28092025-2143
Epoch 15 Loss=0.0061 Batch_id=937 Accuracy=98.76: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:11<00:00, 13.09it/s] 
28092025-2144

Test set: Average loss: 0.0209, Accuracy: 9938/10000 (99.38%)

-----------------------------------------------
Plot saved as training_results-28092025-2144.png
(.venv) PS C:\Raghu\MyLearnings\ERA_V4\S6-21092025\code\mnist_digits_experiments>

"""