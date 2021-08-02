import torch

t = torch.tensor([[[10, 20, 30, 40, 50, 60],[101, 201, 301, 401, 501, 601]],[[1,2,3,4,5,6],[1,2,3,4,5,6]]])

x = torch.tensor([[1.64648, 1.25195, 9.14844, 8.21875]])

print(t[..., 1])

for i, val in enumerate(t):
    print("i = {}".format(i) + " /// val = {}".format(val))


