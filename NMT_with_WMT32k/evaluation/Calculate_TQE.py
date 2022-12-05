import torch
import numpy as np 
from tqe import TQE
from tqdm import tqdm
import re

GPU_NUM = 1 # 원하는 GPU 번호 입력

device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print(torch.cuda.is_available())

f = open('./single/dropout_alpha/model3/hpys.txt', 'r')
outputs = f.readlines()

f1 = open('./single/dropout_alpha/model3/tqe.txt', 'w')
t_sum, count = 0, 0

for o in tqdm(outputs):
    print(o)
    count +=1
    src, hpy, ref = o.split('|')
    hpy = hpy.replace('<eos>', '').replace('Result: ', '')
    src = src.replace('Source: ', '')
    
    # source : 번역 원문
    # target : 기계 번역 문장
    target = []
    source = []

    # Translation Quality Estimator (QE)
    # https://github.com/theRay07/Translation-Quality-Estimator
    target.append(hpy)
    source.append(src)
    model = TQE('LaBSE')
    cos_sim_values = model.fit(source, target)
    t_sum += cos_sim_values[0] 
    
    f1.write(str(cos_sim_values[0]))
    f1.write('\n')
    
    
print('sum: ', t_sum)
print('count: ', count)
print('mean: ', t_sum/count)
f1.write('Average TQE: ' + str(t_sum/count))   

f1.close()
f.close()