# bert score는 bert score repo download 
# https://github.com/Tiiiger/bert_score
# dir = 'bert_score/esb_bertscore.py'

from bert_score import score 
import re

path = '/home/gpuadmin/ahjeong/bert_score/'

cands, refs = [], []
with open(path + "bertscore_hpys/model12_hpys.txt") as f:
  for o in f:
    src, hpy, ref = o.split('|')
    hpy = hpy.replace('<eos>', '').replace('Result: ', '')
    ref = re.sub(r"([?.!,¿])", r" \1 ", ref)
    ref = re.sub(r'[" "]+', " ", ref)
    ref = ref.replace('Target: ', '').replace('\n', '')

    cands.append(hpy)
    refs.append(ref)

P, R, F1 = score(cands, refs, lang="en", verbose=True, device='cuda:0')

f = open(path + 'bertscore_result/model12/P_BERT.txt', 'w')
f1 = open(path + 'bertscore_result/model12/R_BERT.txt', 'w')
f2 = open(path + 'bertscore_result/model12/F1_BERT.txt', 'w')

for i in range(len(F1)):
    f.write(str(P[i].item()))
    f.write('\n')

    f1.write(str(R[i].item()))
    f1.write('\n')

    f2.write(str(F1[i].item()))
    f2.write('\n')

f.write(f"System level F1 score: {P.mean()}")
f1.write(f"System level F1 score: {R.mean()}")
f2.write(f"System level F1 score: {F1.mean()}")

f.close()
f1.close()
f2.close()

