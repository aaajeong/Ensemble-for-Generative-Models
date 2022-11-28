import torch
import numpy as np 
from tqe import TQE
from tqdm import tqdm
import re

f = open('./esb/majority/hpys.txt', 'r')
outputs = f.readlines()

f1 = open('./esb/majority/tqe.txt', 'w')
sum, count = 0, 0

for o in tqdm(outputs):
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
    sum += cos_sim_values[0] 
    
    f1.write(str(cos_sim_values[0]))
    f1.write('\n')
    
print('sum: ', sum)
print('count: ', count)
print('mean: ', sum/count)
f1.write('Average TQE: ' + str(sum/count))   

f1.close()
f.close()