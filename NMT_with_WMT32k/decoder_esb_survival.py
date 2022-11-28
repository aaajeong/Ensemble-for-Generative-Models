import argparse
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

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
    best_token_indices = torch.fmod(best_indices, vocab_size)
    new_batch = torch.index_select(targets, 0, best_tensor_indices)
    new_batch[:, idx] = best_token_indices
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

def get_settled_output(best_scores, best_indices):
    # top k 중 1순위 점수, 인덱스 가져오기
    idx_list, score_list = [], []
    for m in best_indices:
        idx_list.append(m[0])
    for m in best_scores:
        score_list.append(m[0])
    
    # idx_list 중 각 idx가 등장하는 횟수 count
    # max_idx: 가장 많이 등장한 값의 인덱스
    idx_list = torch.tensor(idx_list)
    vals, counts = torch.unique(idx_list, return_counts = True)
    max_idx = torch.where(counts == counts.max())
    
    # idx-scores 딕셔너리 생성
    idx_score = {}
    for i in range(len(idx_list)):
        if idx_list[i].item() not in idx_score:
            idx_score[idx_list[i].item()] = [score_list[i]]
        else:
            idx_score[idx_list[i].item()].append(score_list[i])
    
    # Settled Output 결정
    probs = []
    
    # 최빈값이 여러개면 softmax 값의 평균으로 결정
    scores = []
    if len(max_idx[0]) != 1:
        for i in range(len(max_idx[0])):
            scores.append(torch.mean(torch.tensor(idx_score[vals[max_idx[0][i]].item()])))

        scores = torch.tensor(scores)
        
        result = vals[max_idx[0][torch.argmax(scores)]]
        return result, idx_list
    else:
        result = vals[max_idx[0][0]]
        return result, idx_list
    
        

# python decoder_esb_survival.py --translate --data_dir ./wmt32k_data --model_dir ./outputs --eval_dir ./deu-eng

# dropout은 다른 gpu(id)에서 학습 -> torch.load에서 map_location 설정.
# python decoder_esb_survival.py --translate --data_dir ./wmt32k_data --model_dir ./outputs_dropout --eval_dir ./deu-eng

# dropout & alpha
# python decoder_esb_survival.py --translate --data_dir ./wmt32k_data --model_dir ./outputs_dropout --eval_dir ./deu-eng --alpha_esb

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
    
    # alpha_esb = [0.6, 0.4, 0.2, 0.0, 0.6, 0.4, 0.2, 0.0, 0.8, 1.0]
    alpha_esb = [0.4, 0.2, 0.0, 0.6, 0.4, 0.2, 0.0, 1.0, 0.4, 0.5]  # model 11, 12 추가

    # Load fields.
    if args.translate:
        src_data = torch.load(args.data_dir + '/source.pt')
    trg_data = torch.load(args.data_dir + '/target.pt')

    # Check device cuda 
    device = torch.device('cpu' if args.no_cuda else 'cuda:1')
    
    # Load a saved model.
    origin_models = []
    m = 10
    m2 = 12
    models_dir = args.model_dir
    for i in range(1, m2+1):
        if i == 1:
            continue
        elif i == 9:
            continue
        model_path = models_dir + '/output_' + str(i) + '/last/models'
        model = utils.load_checkpoint(model_path, device)
        origin_models.append(model)
    # m = 10
    # models_dir = args.model_dir
    # for i in range(1, m+1):
    #     model_path = models_dir + '/output_' + str(i) + '/last/models'
    #     model = utils.load_checkpoint(model_path, device)
    #     origin_models.append(model)

    pads = torch.tensor([trg_data['pad_idx']] * beam_size, device=device)
    pads = pads.unsqueeze(-1)

    # We'll find a target sequence by beam search.
    scores_history = [torch.zeros((beam_size,), dtype=torch.float,
                                  device=device)]
    

    eos_idx = trg_data['field'].vocab.stoi[trg_data['field'].eos_token]


    f = open(f'{args.eval_dir}/testset_small.txt', 'r')
    # f = open(f'{args.eval_dir}/oneline.txt', 'r')
    dataset = f.readlines()
    f.close()
    
    
    f = open('./evaluation/esb/survival/dropout_alpha2/hpys.txt', 'w')
    for data in tqdm(dataset):
        # Declare variables for each models
        models = origin_models
        cache = []
        indices_history = []
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

        # 번역 시작
        with torch.no_grad():
            for idx in range(start_idx, args.max_length):
                best_scores = []
                best_indices = []
                
                # 각 모델 encoding 시작
                for i in range(len(models)):
                    if idx > start_idx:
                        targets[i] = torch.cat((targets[i], pads), dim=1)
                        
                    t_self_masks[i] = utils.create_trg_self_mask(targets[i].size()[1],
                                                        device=targets[i].device)
                
                    t_masks[i] = utils.create_pad_mask(targets[i], trg_data['pad_idx'])

                    preds[i] = models[i].decode(targets[i], encoder_outputs[i], src_masks[i],
                                    t_self_masks[i], t_masks[i], cache[i])
                    

                    preds[i] = preds[i][:, idx].squeeze(1)
                    vocab_size = preds[i].size(1)    

                    preds[i] = F.log_softmax(preds[i], dim=1)

                    if idx == start_idx:
                        scores[i] = preds[i][0]

                    else:
                        scores[i] = scores_history[i][-1].unsqueeze(1) + preds[i]

                    # length_penalty = pow(((5. + idx + 1.) / 6.), args.alpha)
                    if args.alpha_esb:
                        length_penalties[i] = pow(((5. + idx + 1.) / 6.), alpha_esb[i])
                        scores[i] = scores[i] / length_penalties[i]
                        scores[i] = scores[i].view(-1)
                    else:
                        length_penalty = pow(((5. + idx + 1.) / 6.), args.alpha)
                        scores[i] = scores[i] / length_penalty
                        scores[i] = scores[i].view(-1)
                
                    # scores[i] = scores[i] / length_penalty
                    # scores[i] = scores[i].view(-1)

                # Ensemble: Survival                
                for i in range(len(models)):    
                    best_score, best_indice = scores[i].topk(beam_size, 0)
                    
                    scores_history[i].append(best_score)
                    indices_history[i].append(best_indice)
                    
                    best_scores.append(best_score)
                    best_indices.append(best_indice)
                
                # Settled Output 결정
                settled, top1_idx = get_settled_output(best_scores, best_indices)
                
                best_token_idx = settled % 38979
                best_token = trg_data['field'].vocab.itos[best_token_idx]
        
                # Winner & Loser 모델 결정
                winner = list(torch.where(top1_idx.clone().detach() == settled)[0])
                loser = list(torch.where(top1_idx.clone().detach() != settled)[0]) 
                '''
                [tensor(0), tensor(1), tensor(2), tensor(3), tensor(4), tensor(5), tensor(6), tensor(7), tensor(8), tensor(9)]
                []
                '''
                
                # 서바이벌 참여 모델 업데이트
                update_sm = []
                u_targets, u_t_self_masks, u_t_masks, u_preds, u_scores = [], [], [], [], []
                u_scores_history, u_indices_history = [], []
                u_best_indices, u_best_scores = [], []
                u_encoder_outputs, u_src_masks, u_cache = [], [], []
                
                for win in winner:
                    update_sm.append(models[win.item()])
                    u_targets.append(targets[win.item()])
                    u_t_self_masks.append(t_self_masks[win.item()])
                    u_t_masks.append(t_masks[win.item()])
                    u_preds.append(preds[win.item()])
                    u_scores.append(scores[win.item()])
                    u_scores_history.append(scores_history[win.item()])
                    u_indices_history.append(indices_history[win.item()])
                    u_best_indices.append(best_indices[win.item()])
                    u_best_scores.append(best_scores[win.item()])
                    u_encoder_outputs.append(encoder_outputs[win.item()])
                    u_src_masks.append(src_masks[win.item()])
                    u_cache.append(cache[win.item()])
                
                models = update_sm
                targets = u_targets
                t_self_masks = u_t_self_masks
                t_masks = u_t_masks
                preds = u_preds
                scores = u_scores
                scores_history = u_scores_history
                indices_history = u_indices_history
                best_indices = u_best_indices
                best_scores = u_best_scores
                encoder_outputs = u_encoder_outputs
                src_masks = u_src_masks
                cache = u_cache
                
                # Stop searching when the best output of beam is EOS.
                if settled.item() % vocab_size == eos_idx:
                    break

                # 각 모델 다음 target update
                for i in range(len(models)):
                    targets[i] = update_targets(targets[i], best_indices[i], idx, vocab_size)
                    
                    

        # Winner 모델의 출력을 가져오기
        result = get_result_sentence(indices_history[0], trg_data, vocab_size)
        f.write("Source: {}|Result: {}|Target: {}\n".format(source, result, target))
                
        # # Winner 전체 모델 결과 출력
        # for i in range(len(models)):
        #     print(get_result_sentence(indices_history[i], trg_data, vocab_size))
    
        
        # f.write("Elapsed Time: {:.2f} sec\n".format(time.time() - start_time))
        # f.write("\n")
        # print("Result: {}".format(result))
        # print("Elapsed Time: {:.2f} sec".format(time.time() - start_time))
        
    f.close()

if __name__ == '__main__':
    main()
