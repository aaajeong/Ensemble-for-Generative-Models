import torch
import sys
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
    project_dir = '/home/nyongja/ahjeong/Ensemble-for-Generative-Models/Neural Machine Translation'
    result_path = f'{project_dir}/outputs/supermodel'
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    f = open(result_path +'/hpys.txt', 'r')
    outputs = f.readlines()
    f.close()
    
    f = open(project_dir + '/dataset/test/source.txt', 'r')
    sources = f.readlines()
    f.close()
    
    f = open(project_dir + '/dataset/test/target.txt', 'r')
    targets = f.readlines()
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
    
    for i in range(len(outputs)):
        if outputs[i] == 'no word\n':
            outputs[i] = ' \n'
        # BLEU 측정 위한 preprocessing
        hpy = outputs[i].replace('<end> ', '')
        src = sources[i]
        ref = targets[i].split()  
        
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
            if hpy == ' \n':
                temp_rouge = rouge.get_scores(' '.join(ref), hpy, avg=True)['rouge-l']
            else:
                temp_rouge = rouge.get_scores(' '.join(ref), ' '.join(hpy.split()), avg=True)['rouge-l']
                
            rouge_l_f1 += temp_rouge['f']
            rouge_l_precision += temp_rouge['p']
            rouge_l_recall += temp_rouge['r']
            all_rouge.append(temp_rouge['f'])
        else:
            print("pass:", ref, glob)
            print(hpy)
            glob_list.append(glob)
            
    
    sbleu = sbleu / len(outputs)
    bleu_1gram = bleu_1gram / len(outputs)
    bleu_2gram = bleu_2gram / len(outputs)
    bleu_3gram = bleu_3gram / len(outputs)
    bleu_4gram = bleu_4gram / len(outputs)
    # meteor = meteor / len(outputs)
    rouge_l_f1 = rouge_l_f1 / len(outputs)
    rouge_l_precision = rouge_l_precision / len(outputs)
    rouge_l_recall = rouge_l_recall / len(outputs) 
    
    cbleu = corpus_bleu(refs, hpys, smoothing_function=smooth.method4)

    f = open(result_path + '/evaluation/hpys_eval.txt', 'a')
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

    f = open(result_path + '/evaluation/hpys_all_bleu.txt', 'a')
    f.write(','.join(map(str,all_bleu)))
    f.close()

    # f = open(result_path + '/all_meteor.txt', 'a')
    # f.write(','.join(map(str,all_meteor)))
    # f.close()
    
    f = open(result_path + '/evaluation/hpys_all_rouge.txt', 'a')
    f.write(','.join(map(str,all_rouge)))
    f.close()
