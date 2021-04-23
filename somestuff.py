import torch
import numpy as numpy

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(ones_tensor.device)


tensor = torch.ones(4, 4)
print("first row ", tensor[0])
print("first column ", tensor[:, 0])
print("last column ", tensor[..., -1])

# join tensors ALONG a dimension
t1 = torch.cat([tensor, tensor, tensor], dim=-2)
print(t1)

# matrix mult
y1 = tensor @ tensor.T
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# element wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)

agg = tensor.sum()
print(agg)

# convert a one-element tensor to python number.
agg_item = agg.item()

# in place operations
tensor.add_(5)

# anything with _ suffix will destructively mutate the called tensor..

# tensors on cpu and numpy arrays can share their underlying memory locations so changing one will change the other.
