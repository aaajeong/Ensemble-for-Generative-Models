eval_dir = "/home/ahjeong/ahjeong/Ensemble-for-Generative-Models/NMT_with_WMT32k/deu-eng"
de = open(f'{eval_dir}/newstest2012.de', 'r')
en = open(f'{eval_dir}/newstest2012.en', 'r')
deset = de.readlines()
enset = en.readlines()

print(len(deset))
print(len(enset))

for i in range(10):
    tgt = enset[i].strip()
    src = deset[i].strip()
    
    print(tgt)
    print(src)