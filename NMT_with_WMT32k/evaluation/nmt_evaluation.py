import torch
import sys
sys.path.append('/home/kie/ahjeong/Ensemble-for-Generative-Models/NMT_with_WMT32k')
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import re

if __name__=="__main__":
    project_dir = '/home/kie/ahjeong/Ensemble-for-Generative-Models/NMT_with_WMT32k'
    result_path = f'{project_dir}/evaluation/single/supermodel'
    # result_path = f'{project_dir}/evaluation/esb/majority/dropout_smoothing'
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    # f = open(result_path +'/hpys_m1.txt', 'r')
    f = open(result_path +'/hpys.txt', 'r')
    output = f.readlines()
    f.close()
    
    sbleu = 0
    bleu_1gram = 0
    bleu_2gram = 0
    bleu_3gram = 0
    bleu_4gram = 0
    meteor = 0
    rouge_l_f1 = 0 
    rouge_l_precision = 0 
    rouge_l_recall = 0 
    
    glob = 0
    glob_list = []
    
    all_bleu = []
    all_meteor = []
    all_rouge = []
    
    refs= []
    hpys = []
    
    smooth = SmoothingFunction()
    
    for o in output:
        
        # BLEU 측정 위한 preprocessing
        # 특정문자열 제거, 단어와 단어 뒤에 오는 구두점(.)사이에 공백을 생성
        src, hpy, ref = o.split('|')
        '''
        src:Source: Hatten Sie Freude daran?
        hpy: Result: did you enjoy it ? <eos>
        ref: Target: Did you enjoy that?
        '''
        
        hpy = hpy.replace('<eos>', '').replace('Result: ', '')
        ref = re.sub(r"([?.!,¿])", r" \1 ", ref)
        ref = re.sub(r'[" "]+', " ", ref)
        ref = ref.replace('Target: ', '').replace('\n', '').split()
        
        refs.append([ref])
        hpys.append(hpy.split())
        
        # 성능 평가
        if len(hpy) > 1:
            sbleu += nltk.translate.bleu(ref, hpy, smoothing_function=smooth.method4)   # weights=(0.25, 0.25, 0.25, 0.25))
            all_bleu.append(nltk.translate.bleu(ref, hpy, smoothing_function=smooth.method4))

            bleu_1gram += sentence_bleu(ref, hpy, weights=(1, 0, 0, 0), smoothing_function=smooth.method4)
            bleu_2gram += sentence_bleu(ref, hpy, weights=(0, 1, 0, 0), smoothing_function=smooth.method4)
            bleu_3gram += sentence_bleu(ref, hpy, weights=(0, 0, 1, 0), smoothing_function=smooth.method4)
            bleu_4gram += sentence_bleu(ref, hpy, weights=(0, 0, 0, 1), smoothing_function=smooth.method4)

            # meteor += meteor_score(ref, hpy)
            # all_meteor.append(meteor_score(ref, hpy))
            
            rouge = Rouge()
            temp_rouge = rouge.get_scores(' '.join(ref), ' '.join(hpy.split()), avg=True)['rouge-l']
            rouge_l_f1 += temp_rouge['f']
            rouge_l_precision += temp_rouge['p']
            rouge_l_recall += temp_rouge['r']
            all_rouge.append(temp_rouge['f'])
        else:
            print("pass:", ref, glob)
            print(hpy)
            glob_list.append(glob)

    
    sbleu = sbleu / len(output)
    bleu_1gram = bleu_1gram / len(output)
    bleu_2gram = bleu_2gram / len(output)
    bleu_3gram = bleu_3gram / len(output)
    bleu_4gram = bleu_4gram / len(output)
    # meteor = meteor / len(output)
    rouge_l_f1 = rouge_l_f1 / len(output)
    rouge_l_precision = rouge_l_precision / len(output)
    rouge_l_recall = rouge_l_recall / len(output) 
    
    cbleu = corpus_bleu(refs, hpys, smoothing_function=smooth.method4)

    f = open(result_path + '/evaluation.txt', 'a')
    f.write('sentence bleu: '+ str(sbleu)+'\n')
    f.write('corpus bleu: '+ str(cbleu)+'\n')
    f.write('1-Gram BLEU: '+ str(bleu_1gram)+'\n')
    f.write('2-Gram BLEU: '+ str(bleu_2gram)+'\n')
    f.write('3-Gram BLEU: '+ str(bleu_3gram)+'\n')
    f.write('4-Gram BLEU: '+ str(bleu_4gram)+'\n')
    # f.write('METEOR: '+ str(meteor)+'\n')
    f.write('ROUGE-L F1 score: '+ str(rouge_l_f1)+'\n')
    f.write('ROUGE-L Precision: '+ str(rouge_l_precision)+'\n')
    f.write('ROUGE-L Recall: '+ str(rouge_l_recall)+'\n')
    f.close()

    f = open(result_path + '/all_bleu.txt', 'a')
    f.write(','.join(map(str,all_bleu)))
    f.close()

    # f = open(result_path + '/all_meteor.txt', 'a')
    # f.write(','.join(map(str,all_meteor)))
    # f.close()
    
    f = open(result_path + '/all_rouge.txt', 'a')
    f.write(','.join(map(str,all_rouge)))
    f.close()