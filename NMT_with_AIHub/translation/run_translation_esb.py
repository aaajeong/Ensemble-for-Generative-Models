import warnings

from numpy import source
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append('/home/kie/ahjeong/pytorch-transformer')
import pandas as pd
import torch
from torch.autograd import Variable
from model.util import subsequent_mask
from model.transformer import Transformer
from transformers import BertTokenizer
import datetime 
from tqdm import tqdm

if __name__=="__main__":
  project_dir = '/home/kie/ahjeong/pytorch-transformer'
  vocab_path = f'{project_dir}/data/vocab_short.txt'
  data_path = f'{project_dir}/data/kor_eng_test2.csv'
  checkpoint_path = f'{project_dir}/checkpoints'
  output_dir = os.path.join(
        f'{project_dir}/result/test/', datetime.datetime.now().strftime('%Y-%m-%d-%H:%M'), 'esb')  # _%H-%M-%S
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
    
  # model setting
  models_name = ['transformer-translation-spoken', 'transformer-translation-spoken4']
  models = []
  checkpoints = []
  start_epochs = []
  losses_list = []
  global_steps_list = []
  
  vocab_num = 32000
  max_length = 512  #64
  d_model = 512
  head_num = 8
  dropout = 0.1
  N = 6
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

  tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)
  
  for _ in range(len(models_name)):
    model = Transformer(vocab_num=vocab_num,
                      d_model=d_model,
                      max_seq_len=max_length,
                      head_num=head_num,
                      dropout=dropout,
                      N=N)
    models.append(model)
  
  if models_name:
    for m in range(len(models)):
      checkpoint = torch.load(f'{checkpoint_path}/{models_name[m]}.pth', map_location=device)
      start_epoch = checkpoint['epoch']
      losses = checkpoint['losses']
      global_steps = checkpoint['train_step']
      
      # 각 리스트에 추가
      checkpoints.append(checkpoint)
      start_epochs.append(start_epoch)
      losses_list.append(losses)
      global_steps_list.append(global_steps)
      
      models[m].load_state_dict(checkpoints[m]['model_state_dict'])
      print(f'{checkpoint_path}/{models_name[m]}-.pth loaded')
      
      models[m].eval()


  test_data = pd.read_csv(data_path, encoding = 'utf-8')
  src = list(test_data['원문'])
  tgt = list(test_data['번역문'])
  
  # # Transformer 모델 별 변수 list 생성
  # encoder_inputs = []
  # encoder_masks = []
  # targets = []
  # encoder_outputs = []
  
  f = open(output_dir + '/output.txt', 'a')
  for i in tqdm(range(len(src))):
     # Transformer 모델 별 변수 list 생성
    encoder_inputs = []
    encoder_masks = []
    targets = []
    encoder_outputs = []
    # input_str = '10,000원짜리 지폐 1장을 빌려주시겠어요?'
    input_str = src[i]
    target_str = tgt[i]
    
    str = tokenizer.encode(input_str)
    
    pad_len = (max_length - len(str))
    str_len = len(str)
    
    for m in range(len(models)):
      encoder_inputs.append(torch.tensor(str + [tokenizer.pad_token_id]* pad_len))
      encoder_masks.append((encoder_inputs[m] != tokenizer.pad_token_id).unsqueeze(0))
      targets.append(torch.ones(1, 1).fill_(tokenizer.cls_token_id).type_as(encoder_inputs[m]))
      encoder_outputs.append(models[m].encode(encoder_inputs[m],encoder_masks[m]))
  
    
    for i in range(max_length - 1):
    
      lm_logits_list = []
      probs = []
      next_words = []
      # 각 모델 decode 
      for m in range(len(models)):
        lm_logits_list.append(models[m].decode(encoder_outputs[m],encoder_masks[m],targets[m], Variable(subsequent_mask(targets[m].size(1)).type_as(encoder_inputs[m].data))))
        probs.append(lm_logits_list[m][:, -1])
      
      # Ensemble: Soft Voting
      prob_esb = torch.add(probs[0], probs[1])
      _, esb_word = torch.max(prob_esb, dim=1)
      
      # for m in range(len(models)):
      #   _, next_word = torch.max(probs[m], dim=1)
      #   next_words.append(next_word)
        
      # esb word를 가 모델 next word로.
      for m in range(len(models)):
        next_words.append(esb_word.clone().detach())

      # print(f'ko: {input_str} en: {tokenizer.decode(target.squeeze().tolist(), skip_special_tokens=True)}')
      
      # esb_word 가 [pad], [sep] 인지 확인
      if esb_word.data[0] == tokenizer.pad_token_id or esb_word.data[0] == tokenizer.sep_token_id:
        # print(f'ko: {input_str} en: {tokenizer.decode(targets[0].squeeze().tolist(),skip_special_tokens=True)}')
        output = tokenizer.decode(targets[0].squeeze().tolist(),skip_special_tokens=True)
        f.write(output + '|' + target_str + '\n')
        break
      
      for m in range(len(models)):
        targets[m] = torch.cat((targets[m][0], esb_word))
        targets[m] = targets[m].unsqueeze(0)
        
  f.close
  
  # # Transformer 모델 별 변수 list 생성
  # encoder_inputs = []
  # encoder_masks = []
  # targets = []
  # encoder_outputs = []
  
  
  # while True:
    
  #   input_str = '10,000원짜리 지폐 1장을 빌려주시겠어요?'
  #   str = tokenizer.encode(input_str)
    
  #   pad_len = (max_length - len(str))
  #   str_len = len(str)
    
  #   for m in range(len(models)):
  #     encoder_inputs.append(torch.tensor(str + [tokenizer.pad_token_id]* pad_len))
  #     encoder_masks.append((encoder_inputs[m] != tokenizer.pad_token_id).unsqueeze(0))
  #     targets.append(torch.ones(1, 1).fill_(tokenizer.cls_token_id).type_as(encoder_inputs[m]))
  #     encoder_outputs.append(models[m].encode(encoder_inputs[m],encoder_masks[m]))
    
  #   for i in range(max_length - 1):
    
  #     lm_logits_list = []
  #     probs = []
  #     next_words = []
  #     # 각 모델 decode 
  #     for m in range(len(models)):
  #       lm_logits_list.append(models[m].decode(encoder_outputs[m],encoder_masks[m],targets[m], Variable(subsequent_mask(targets[m].size(1)).type_as(encoder_inputs[m].data))))
  #       probs.append(lm_logits_list[m][:, -1])
      
  #     # Ensemble: Soft Voting
  #     prob_esb = torch.add(probs[0], probs[1])
  #     _, esb_word = torch.max(prob_esb, dim=1)
  #     print('esb_word: ', esb_word)
      
  #     # for m in range(len(models)):
  #     #   _, next_word = torch.max(probs[m], dim=1)
  #     #   next_words.append(next_word)
        
  #     # esb word를 가 모델 next word로.
  #     for m in range(len(models)):
  #       next_words.append(esb_word.clone().detach())
      
  #     print('next_word: ', next_words[0])
  #     print('next_word2: ', next_words[1])
  #     print('\n')
  #     # print(f'ko: {input_str} en: {tokenizer.decode(target.squeeze().tolist(), skip_special_tokens=True)}')
      
  #     # esb_word 가 [pad], [sep] 인지 확인
  #     if esb_word.data[0] == tokenizer.pad_token_id or esb_word.data[0] == tokenizer.sep_token_id:
  #       print(f'ko: {input_str} en: {tokenizer.decode(targets[0].squeeze().tolist(),skip_special_tokens=True)}')
  #       break
      
  #     for m in range(len(models)):
  #       targets[m] = torch.cat((targets[m][0], esb_word))
  #       targets[m] = targets[m].unsqueeze(0)  
      
  #   break