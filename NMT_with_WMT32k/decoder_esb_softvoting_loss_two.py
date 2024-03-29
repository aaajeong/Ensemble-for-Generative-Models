import argparse
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import utils

# pylint: disable=not-callable


def encode_inputs(sentence, model, src_data, beam_size, device):
    inputs = src_data['field'].preprocess(sentence)
    inputs.append(src_data['field'].eos_token)
    inputs = [inputs]
    inputs = src_data['field'].process(inputs, device=device)
    with torch.no_grad():
        src_mask = utils.create_pad_mask(inputs, src_data['pad_idx'])
        enc_output = model.encode(inputs, src_mask)
        enc_output = enc_output.repeat(beam_size, 1, 1)
    return enc_output, src_mask


def update_targets(targets, best_indices, idx, vocab_size):
    best_tensor_indices = torch.div(best_indices, vocab_size)
    # print('bext_tesor_indices: ', best_tensor_indices)
    best_token_indices = torch.fmod(best_indices, vocab_size)
    # print('best_token_indices: ', best_token_indices)
    # print('targets: ', targets)
    new_batch = torch.index_select(targets, 0, best_tensor_indices)
    # print('new_batch 전: ', new_batch)
    new_batch[:, idx] = best_token_indices
    # print('new_batch 후: ', new_batch)
    return new_batch


def get_result_sentence(indices_history, trg_data, vocab_size):
    result = []
    k = 0
    for best_indices in indices_history[::-1]:
        best_idx = best_indices[k]
        # TODO: get this vocab_size from target.pt?
        k = best_idx // vocab_size
        best_token_idx = best_idx % vocab_size
        best_token = trg_data['field'].vocab.itos[best_token_idx]
        result.append(best_token)
        
    return ' '.join(result[::-1])

# python decoder_esb_softvoting.py --translate --data_dir ./wmt32k_data --model_dir ./outputs --eval_dir ./deu-eng

# dropout은 다른 gpu(id)에서 학습 -> torch.load에서 map_location 설정.
# python decoder_esb_softvoting.py --translate --data_dir ./wmt32k_data --model_dir ./outputs_dropout --eval_dir ./deu-eng

# dropout & alpha
# nohup python decoder_esb_softvoting_loss_two.py --translate --data_dir ./wmt32k_data --model_dir ./outputs_dropout --eval_dir ./deu-eng --alpha_esb &

# dropout & label smoothing
# python decoder_esb_softvoting.py --translate --data_dir ./wmt32k_data --model_dir ./outputs_smoothing --eval_dir ./deu-eng

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--eval_dir', type=str, required=False)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--alpha_esb', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--translate', action='store_true')
    
    args = parser.parse_args()

    beam_size = args.beam_size
    
    alpha_esb = [0.0, 0.0]  # model 4, model 12

    # MAX - x / MAX-MIN
    loss_esb = [1, 0]   # model 4, model 12


    # Load fields.
    if args.translate:
        src_data = torch.load(args.data_dir + '/source.pt')
    trg_data = torch.load(args.data_dir + '/target.pt')

    # Check device cuda 
    device = torch.device('cpu' if args.no_cuda else 'cuda:0')
    
    # Load a saved model.
    models = []
    m = 2
    models_dir = args.model_dir
    
    # Model 4
    model_path = models_dir + '/output_' + str(4) + '/last/models'
    model = utils.load_checkpoint2(model_path, device)
    models.append(model)
    
    # Model 12
    model_path = models_dir + '/output_' + str(4) + '/last/models'
    model = utils.load_checkpoint2(model_path, device)
    models.append(model)
    
    pads = torch.tensor([trg_data['pad_idx']] * beam_size, device=device)
    pads = pads.unsqueeze(-1)

    # We'll find a target sequence by beam search.
    scores_history = [torch.zeros((beam_size,), dtype=torch.float,
                                  device=device)]
    

    eos_idx = trg_data['field'].vocab.stoi[trg_data['field'].eos_token]


    # f = open(f'{args.eval_dir}/testset_small.txt', 'r')
    f = open(f'{args.eval_dir}/oneline.txt', 'r')
    # f = open(f'{args.eval_dir}/oneline2.txt', 'r')
    dataset = f.readlines()
    f.close()
    
    
    f = open('./evaluation/valid/esb/1:0_x/hpys.txt', 'w')
    for data in tqdm(dataset):
        # Declare variables for each models
        cache = []
        indices_history = []
        m4_indices_history = []
        scores_history = []
        encoder_outputs = []
        src_masks = []
        targets = []
        t_self_masks = []
        t_masks = []
        preds = []
        scores = []
        length_penalties = []
        
        for _ in range(m):
            cache.append({})
            indices_history.append([])
            
            # We'll find a target sequence by beam search.
            scores_history.append([torch.zeros((beam_size,), dtype=torch.float,
                                    device=device)])
            
            # target self mask, mask
            t_self_masks.append(None)
            t_masks.append(None)
            
            # prediction
            preds.append(None)
            
            # score
            scores.append(None)
            
            # length_penalty
            length_penalties.append(None)
            
        target, source = data.strip().split('\t')   # 원래 데이터셋 형태: en -> de
        
        if args.translate:
            # sentence = input('Source? ')
            sentence = source   # de
        
        # Encoding inputs.
        if args.translate:
            start_time = time.time()
            for i in range(m):
                enc_output, src_mask = encode_inputs(sentence, models[i], src_data,
                                                beam_size, device)
                encoder_outputs.append(enc_output)
                src_masks.append(src_mask)
                targets.append(pads)
            
            start_idx = 0
        else:
            for i in range(m):    
                enc_output, src_mask = None, None
                encoder_outputs.append(enc_output)
                src_masks.append(src_mask)
                targets.append(torch.tensor([sentence], device=device))
                
            sentence = input('Target? ').split()
            for idx, _ in enumerate(sentence):
                sentence[idx] = trg_data['field'].vocab.stoi[sentence[idx]]
            sentence.append(trg_data['pad_idx'])
            
            start_idx = targets[0].size(1) - 1
            start_time = time.time()
        
        # === Encoder output & source mask & taret 확인 ===
        # for i in range(m):
        #     print(i , ' 번 모델 Encoder output: ',encoder_outputs[i])
        #     print(i , ' 번 모델 source mask: ',src_masks[i])
        #     print(i , ' 번 모델 taret: ',targets[i])

        # 번역 시작
        with torch.no_grad():
            for idx in range(start_idx, args.max_length):
                
                # 각 모델 encoding 시작
                for i in range(m):
                    if idx > start_idx:
                        targets[i] = torch.cat((targets[i], pads), dim=1)
                        # print(i, ' 번 모델 targets: ', targets[i])
                        
                    t_self_masks[i] = utils.create_trg_self_mask(targets[i].size()[1],
                                                        device=targets[i].device)
                    # print(i, ' 번 모델 t_self_masks: ', t_self_masks[i])
                
                    t_masks[i] = utils.create_pad_mask(targets[i], trg_data['pad_idx'])
                    # print(i, ' 번 모델 t_masks: ', t_masks[i])

                    preds[i] = models[i].decode(targets[i], encoder_outputs[i], src_masks[i],
                                    t_self_masks[i], t_masks[i], cache[i])
                    # print(i, ' 번 모델 cache: ', cache[i])
                    # print(i, ' 번 모델 preds: ', preds[i])
                    

                    preds[i] = preds[i][:, idx].squeeze(1)
                    # print(i, ' 번 모델 squeeze preds: ', preds[i])
                    vocab_size = preds[i].size(1)    

                    preds[i] = F.log_softmax(preds[i], dim=1)
                    # print(i, ' 번 모델 softmax preds: ', preds[i])

                    if idx == start_idx:
                        scores[i] = preds[i][0]

                    else:
                        scores[i] = scores_history[i][-1].unsqueeze(1) + preds[i]
                        scores[i] = preds[i]
                    # print(i, ' 번 모델 scores: ', scores[i])
                    # length_penalty = pow(((5. + idx + 1.) / 6.), args.alpha)
                
                    # scores[i] = scores[i] / length_penalty
                    # scores[i] = scores[i].view(-1)
                    
                    if args.alpha_esb:
                        length_penalties[i] = pow(((5. + idx + 1.) / 6.), alpha_esb[i])
                        scores[i] = scores[i] / length_penalties[i]
                        scores[i] = scores[i].view(-1)
                        # print(i, ' 번 모델 length penaltiest 후 scores: ', scores[i])
                        
                    else:
                        # print('============= check =============')
                        length_penalty = pow(((5. + idx + 1.) / 6.), args.alpha)
                        scores[i] = scores[i] / length_penalty
                        scores[i] = scores[i].view(-1)
              
                # LOSS 추가
                for i in range(m):
                    # print(i, ' 번째 모델 loss 전 score: ', scores[i])
                    scores[i] = scores[i] * loss_esb[i]
                    # print(i, ' 번째 모델 loss 후 score: ', scores[i])
                
                # Ensemble: Soft Voting
                for i in range(m):
                    if i==0:
                        scores_esb = scores[i]
                    else:
                        scores_esb = torch.add(scores_esb, scores[i])
                # print('soft voting 전 전체 Scores 합: ', scores_esb)
                scores_esb = torch.div(scores_esb, m)
                # print('soft voting 후 전체 Scores 평균: ', scores_esb)

                m4_indices = scores[0].topk(beam_size, 0)
                m4_indices_history.append(m4_indices)
                # 각 모델 best_score, best_indcices 구하기 -> 앞서 앙상블 한 결과이기 때문에 모든 모델은 동일한 결과를 갖게됨.
                for i in range(m):    
                    best_scores, best_indices = scores_esb.topk(beam_size, 0)
                    # print('ensemble 후 topk index: ', best_indices)
                    scores_history[i].append(best_scores)
                    indices_history[i].append(best_indices)
                    

                # Stop searching when the best output of beam is EOS.
                if best_indices[0].item() % vocab_size == eos_idx:
                    break
                
                # 각 모델 다음 target update
                for i in range(m):
                    # print(i, ' 번 모델 update 전 Targets: ', targets[i])
                    targets[i] = update_targets(targets[i], best_indices, idx, vocab_size)
                    # print(i, ' 번 모델 update 된 Targets: ', targets[i])
        # print('model 4 none ensemble index: ', m4_indices_history)
        # print('model 4 indices_history: ', indices_history[0])
        # print('model 12 indices_history: ', indices_history[1])

        # 모든 모델 같은 출력을 내기 때문에 0번째 모델의 결과를 출력
        result = get_result_sentence(indices_history[0], trg_data, vocab_size)
        f.write("Source: {}|Result: {}|Target: {}\n".format(source, result, target))

    f.close()

if __name__ == '__main__':
    main()
