import torch

settled = torch.tensor(39057)
idx_list = [torch.tensor(39058, device='cuda:1'), torch.tensor(39057, device='cuda:1'), 
            torch.tensor(39057, device='cuda:1'), torch.tensor(39057, device='cuda:1'), 
            torch.tensor(39057, device='cuda:1'), torch.tensor(39057, device='cuda:1'), 
            torch.tensor(39057, device='cuda:1'), torch.tensor(39057, device='cuda:1'), 
            torch.tensor(39057, device='cuda:1'), torch.tensor(39057, device='cuda:1')]


winner = list(torch.where(torch.tensor(idx_list) == settled)[0])
loser = list(torch.where(torch.tensor(idx_list) != settled)[0]) 

print(winner)
print(loser)

for win in loser:
    del idx_list[win.item()]
    
print(idx_list)
print(len(idx_list))