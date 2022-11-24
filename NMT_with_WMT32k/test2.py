import torch 


result = [torch.tensor(-0.7018, device='cuda:1'), torch.tensor(-1.0115, device='cuda:1'), torch.tensor(-1.4247, device='cuda:1'), torch.tensor(-1.9559, device='cuda:1'), torch.tensor(-1.8093, device='cuda:1'), torch.tensor(-1.9736, device='cuda:1'), torch.tensor(-1.2388, device='cuda:1'), torch.tensor(-1.2028, device='cuda:1'), torch.tensor(0., device='cuda:1')]


print(type(torch.tensor(result)))
print(torch.mean(torch.tensor(result)))