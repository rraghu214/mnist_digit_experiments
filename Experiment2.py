# SCROLL DOWN FOR LOGS

from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from datetime import datetime
from Model2 import Net, get_optimizer_and_scheduler


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
    # test_losses.append(test_loss)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

    # test_acc.append(100. * correct / len(test_loader.dataset))

    
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
## 1. Model_2.py logs:
```
(.venv) PS C:\Raghu\MyLearnings\ERA_V4\S6-21092025\code\mnist_digits_experiments> python .\Experiment2.py
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
            Conv2d-9             [-1, 32, 7, 7]           4,640
      BatchNorm2d-10             [-1, 32, 7, 7]              64
           Conv2d-11             [-1, 10, 7, 7]             330
AdaptiveAvgPool2d-12             [-1, 10, 1, 1]               0
================================================================
Total params: 8,682
Trainable params: 8,682
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.39
Params size (MB): 0.03
Estimated Total Size (MB): 0.43
----------------------------------------------------------------
EPOCH: 1
28092025-1423
Loss=0.36920347809791565 Batch_id=468 Accuracy=71.10: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:27<00:00, 17.35it/s]
28092025-1424

Test set: Average loss: 0.4277, Accuracy: 8900/10000 (89.00%)

C:\Raghu\MyLearnings\ERA_V4\S6-21092025\code\mnist_digits_experiments\.venv\Lib\site-packages\torch\optim\lr_scheduler.py:240: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
-----------------------------------------------
EPOCH: 2
28092025-1424
Loss=0.15809984505176544 Batch_id=468 Accuracy=95.14: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:28<00:00, 16.58it/s] 
28092025-1424

Test set: Average loss: 0.1636, Accuracy: 9553/10000 (95.53%)

-----------------------------------------------
EPOCH: 3
28092025-1424
Loss=0.10418612509965897 Batch_id=468 Accuracy=96.42: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:29<00:00, 15.97it/s] 
28092025-1425

Test set: Average loss: 0.0978, Accuracy: 9710/10000 (97.10%)

-----------------------------------------------
EPOCH: 4
28092025-1425
Loss=0.10030681639909744 Batch_id=468 Accuracy=97.03: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:30<00:00, 15.14it/s] 
28092025-1425

Test set: Average loss: 0.0975, Accuracy: 9723/10000 (97.23%)

-----------------------------------------------
EPOCH: 5
28092025-1425
Loss=0.17494289577007294 Batch_id=468 Accuracy=97.43: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:32<00:00, 14.38it/s] 
28092025-1426

Test set: Average loss: 0.0895, Accuracy: 9739/10000 (97.39%)

-----------------------------------------------
EPOCH: 6
28092025-1426
Loss=0.07066521048545837 Batch_id=468 Accuracy=97.53: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:33<00:00, 13.97it/s] 
28092025-1426

Test set: Average loss: 0.0812, Accuracy: 9754/10000 (97.54%)

-----------------------------------------------
EPOCH: 7
28092025-1427
Loss=0.06886536628007889 Batch_id=468 Accuracy=97.81: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:33<00:00, 13.86it/s] 
28092025-1427

Test set: Average loss: 0.0742, Accuracy: 9769/10000 (97.69%)

-----------------------------------------------
EPOCH: 8
28092025-1427
Loss=0.10225506871938705 Batch_id=468 Accuracy=97.97: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:34<00:00, 13.63it/s] 
28092025-1428

Test set: Average loss: 0.0845, Accuracy: 9750/10000 (97.50%)

-----------------------------------------------
EPOCH: 9
28092025-1428
Loss=0.13639578223228455 Batch_id=468 Accuracy=98.05: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:35<00:00, 13.39it/s] 
28092025-1428

Test set: Average loss: 0.0700, Accuracy: 9793/10000 (97.93%)

-----------------------------------------------
EPOCH: 10
28092025-1428
Loss=0.04578019306063652 Batch_id=468 Accuracy=98.17: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:35<00:00, 13.20it/s] 
28092025-1429

Test set: Average loss: 0.0671, Accuracy: 9808/10000 (98.08%)

-----------------------------------------------
EPOCH: 11
28092025-1429
Loss=0.05619727075099945 Batch_id=468 Accuracy=98.23: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.93it/s] 
28092025-1430

Test set: Average loss: 0.0511, Accuracy: 9845/10000 (98.45%)

-----------------------------------------------
EPOCH: 12
28092025-1430
Loss=0.1221398189663887 Batch_id=468 Accuracy=98.28: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.80it/s] 
28092025-1430

Test set: Average loss: 0.0513, Accuracy: 9840/10000 (98.40%)

-----------------------------------------------
EPOCH: 13
28092025-1430
Loss=0.028884632512927055 Batch_id=468 Accuracy=98.46: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:37<00:00, 12.46it/s] 
28092025-1431

Test set: Average loss: 0.0604, Accuracy: 9810/10000 (98.10%)

-----------------------------------------------
EPOCH: 14
28092025-1431
Loss=0.044170115143060684 Batch_id=468 Accuracy=98.43: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:37<00:00, 12.53it/s] 
28092025-1432

Test set: Average loss: 0.0693, Accuracy: 9784/10000 (97.84%)

-----------------------------------------------
EPOCH: 15
28092025-1432
Loss=0.0428624302148819 Batch_id=468 Accuracy=98.55: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:40<00:00, 11.60it/s] 
28092025-1432

Test set: Average loss: 0.0419, Accuracy: 9870/10000 (98.70%)

-----------------------------------------------
Plot saved as training_results-28092025-1433.png
(.venv) PS C:\Raghu\MyLearnings\ERA_V4\S6-21092025\code\mnist_digits_experiments> 
```

"""