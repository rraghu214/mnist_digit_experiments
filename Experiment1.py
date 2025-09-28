# SCROLL DOWN FOR LOGS

from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from datetime import datetime
from Model1 import Net, get_optimizer_and_scheduler


# Train Phase transformations
train_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
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
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)


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


def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
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
    train_losses.append(loss.item())

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

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

    
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({100. * correct / len(test_loader.dataset):.2f}%)\n")

    return test_loss   



model =  Net().to(device)
optimizer, scheduler = get_optimizer_and_scheduler(model)

if __name__ == "__main__":
    EPOCHS = 15
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch+1)
        print(datetime.now().strftime("%d%m%Y-%H%M"))
        train(model, device, train_loader, optimizer, epoch)
        print(datetime.now().strftime("%d%m%Y-%H%M"))
        val_loss=test(model, device, test_loader)
        scheduler.step(val_loss)
        print("-----------------------------------------------")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc[4000:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    
    # Format timestamp as DDMMYYYY-HHMM
    
    timestamp = datetime.now().strftime("%d%m%Y-%H%M")

    # Save with timestamp in filename
    filename = f"training_results-{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    print(f"Plot saved as {filename}")
    # plt.show()

# LOGS

"""

## 1. Model_1.py logs:

(.venv) PS C:\Raghu\MyLearnings\ERA_V4\S6-21092025\code\mnist_digits_experiments> python .\Experiment1.py
CUDA Available? True
cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
         MaxPool2d-3            [-1, 8, 14, 14]               0
            Conv2d-4           [-1, 16, 14, 14]           1,168
       BatchNorm2d-5           [-1, 16, 14, 14]              32
         MaxPool2d-6             [-1, 16, 7, 7]               0
            Conv2d-7              [-1, 8, 7, 7]             136
            Conv2d-8             [-1, 32, 7, 7]           2,336
            Conv2d-9             [-1, 28, 7, 7]             924
           Linear-10                   [-1, 10]          13,730
================================================================
Total params: 18,422
Trainable params: 18,422
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.19
Params size (MB): 0.07
Estimated Total Size (MB): 0.26
----------------------------------------------------------------
EPOCH: 1
28092025-1439
Loss=0.09395576268434525 Batch_id=468 Accuracy=90.50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:32<00:00, 14.52it/s] 
28092025-1439

Test set: Average loss: 0.1156, Accuracy: 9621/10000 (96.21%)

C:\Raghu\MyLearnings\ERA_V4\S6-21092025\code\mnist_digits_experiments\.venv\Lib\site-packages\torch\optim\lr_scheduler.py:240: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
-----------------------------------------------
EPOCH: 2
28092025-1439
Loss=0.09951924532651901 Batch_id=468 Accuracy=97.21: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.77it/s] 
28092025-1440

Test set: Average loss: 0.0483, Accuracy: 9845/10000 (98.45%)

-----------------------------------------------
EPOCH: 3
28092025-1440
Loss=0.02210957370698452 Batch_id=468 Accuracy=97.92: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.89it/s] 
28092025-1441

Test set: Average loss: 0.0381, Accuracy: 9875/10000 (98.75%)

-----------------------------------------------
EPOCH: 4
28092025-1441
Loss=0.062317878007888794 Batch_id=468 Accuracy=98.27: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:42<00:00, 11.08it/s] 
28092025-1441

Test set: Average loss: 0.0412, Accuracy: 9862/10000 (98.62%)

-----------------------------------------------
EPOCH: 5
28092025-1441
Loss=0.01374585647135973 Batch_id=468 Accuracy=98.54: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:46<00:00, 10.19it/s] 
28092025-1442

Test set: Average loss: 0.0491, Accuracy: 9835/10000 (98.35%)

-----------------------------------------------
EPOCH: 6
28092025-1442
Loss=0.04199260100722313 Batch_id=468 Accuracy=98.67: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:52<00:00,  8.88it/s] 
28092025-1443

Test set: Average loss: 0.0387, Accuracy: 9873/10000 (98.73%)

-----------------------------------------------
EPOCH: 7
28092025-1443
Loss=0.021661289036273956 Batch_id=468 Accuracy=98.78: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:58<00:00,  7.99it/s] 
28092025-1444

Test set: Average loss: 0.0298, Accuracy: 9894/10000 (98.94%)

-----------------------------------------------
EPOCH: 8
28092025-1444
Loss=0.032435160130262375 Batch_id=468 Accuracy=98.90: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:53<00:00,  8.77it/s] 
28092025-1445

Test set: Average loss: 0.0358, Accuracy: 9887/10000 (98.87%)

-----------------------------------------------
EPOCH: 9
28092025-1445
Loss=0.02156253717839718 Batch_id=468 Accuracy=98.92: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:53<00:00,  8.69it/s] 
28092025-1446

Test set: Average loss: 0.0249, Accuracy: 9911/10000 (99.11%)

-----------------------------------------------
EPOCH: 10
28092025-1446
Loss=0.03644905239343643 Batch_id=468 Accuracy=99.00: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:56<00:00,  8.31it/s] 
28092025-1447

Test set: Average loss: 0.0314, Accuracy: 9902/10000 (99.02%)

-----------------------------------------------
EPOCH: 11
28092025-1447
Loss=0.009006721898913383 Batch_id=468 Accuracy=99.10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:20<00:00,  5.85it/s] 
28092025-1449

Test set: Average loss: 0.0279, Accuracy: 9905/10000 (99.05%)

-----------------------------------------------
EPOCH: 12
28092025-1449
Loss=0.002540260786190629 Batch_id=468 Accuracy=99.11: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:04<00:00,  7.25it/s] 
28092025-1450

Test set: Average loss: 0.0267, Accuracy: 9912/10000 (99.12%)

-----------------------------------------------
EPOCH: 13
28092025-1450
Loss=0.008956954814493656 Batch_id=468 Accuracy=99.11: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:04<00:00,  7.30it/s] 
28092025-1451

Test set: Average loss: 0.0268, Accuracy: 9905/10000 (99.05%)

-----------------------------------------------
EPOCH: 14
28092025-1451
Loss=0.02956872247159481 Batch_id=468 Accuracy=99.24: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:00<00:00,  7.81it/s] 
28092025-1452

Test set: Average loss: 0.0309, Accuracy: 9904/10000 (99.04%)

-----------------------------------------------
EPOCH: 15
28092025-1452
Loss=0.03176385536789894 Batch_id=468 Accuracy=99.28: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:44<00:00, 10.48it/s] 
28092025-1453

Test set: Average loss: 0.0265, Accuracy: 9916/10000 (99.16%)

-----------------------------------------------
Plot saved as training_results-28092025-1453.png
(.venv) PS C:\Raghu\MyLearnings\ERA_V4\S6-21092025\code\mnist_digits_experiments> 

"""