import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,

    # this transforms the features
    transform=ToTensor(),
    
    # ToTensor converts a PIL image or NumPy nd array into a floattensor and scales the image pixel intensity from 0 to 1

    # this transforms the output

    # so if we have output 1
    # we transform the output into 
    # [0 1 0 0 0 0 0 0]
    # or if we have 5
    # we get [0 0 0 0 1 0 0 0 ]
    # etc
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

x = torch.zeros(10).scatter_(dim=0, index=torch.tensor(1), value = 1)

print(x)