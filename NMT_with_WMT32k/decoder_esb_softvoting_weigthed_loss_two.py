import argparse
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import utils
from collections import defaultdict

# pylint: disable=not-callable

# 같은 단어 답한 모델의 weighted loss 반영 +  분모(weighted 합)

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

def get_settled_topk(best_scores, best_indices, beam_size = 4):
    """
    Hard Voting으로 다수결 단어 선택 -> Settled output 결정
    """
    # top k 점수, 인덱스 가져오기
    best_indices = torch.transpose(best_indices, 0, 1)
    best_scores = torch.transpose(best_scores, 0, 1)
    
    vals = []   # vocab 번호
    counts = [] # vocab 빈도 수
    max_idx = [] # 최빈 vocab 인덱스
    topk_dic = []   # topk의 idx-score 딕셔너리
    topk_result = [] # topk 최종 1등
    for i in range(beam_size):
        val, count = torch.unique(best_indices[i], return_counts = True)
        vals.append(val)
        counts.append(count)
        max_idx.append(torch.where(count == count.max()))
    
    
        # idx-scores 딕셔너리
        idx_score = {}
        for j in range(len(best_indices[i])):
            if best_indices[i][j].item() not in idx_score:
                idx_score[best_indices[i][j].item()] = [best_scores[i][j]]
            else:
                idx_score[best_indices[i][j].item()].append(best_scores[i][j])
        topk_dic.append(idx_score)
    
    for i in range(beam_size):
        # Settled Output 결정
        # 최빈값이 여러개면 softmax 값의 평균으로 결정
        scores = []
        if len(max_idx[i][0]) != 1:
            for j in range(len(max_idx[i][0])):
                scores.append(torch.mean(torch.tensor(topk_dic[i][vals[i][max_idx[i][0][j]].item()])))

            scores = torch.tensor(scores)
            
            result = vals[i][max_idx[i][0][torch.argmax(scores)]]
            topk_result.append(result)
        else:
            result = vals[i][max_idx[i][0][0]]
            topk_result.append(result.item())
    return topk_result

def get_settled_topk_with_index(loss_esbs, best_scores, best_indices, beam_size = 4):
    """
    Hard Voting으로 다수결 단어 선택 -> Settled output 결정
    """
    # top k 점수, 인덱스 가져오기
    best_indices = torch.transpose(best_indices, 0, 1)
    best_scores = torch.transpose(best_scores, 0, 1)
    
    print('best_scores: ', best_scores)
    print('best_indices: ', best_indices)
    # Loss 반영
    best_scores = torch.mul(best_scores, loss_esbs) # Loss 반영
    
    topk_dic = []   # topk의 idx-score 딕셔너리
    topk_result = [] # topk 최종 1등
    idx_score = {}
    idx_model = {}

    for i in range(beam_size):
        print('best_indices[i]: ', best_scores[i])
        for j in range(len(best_indices[i])):
            print('best_indices[i][j].item(): ', best_indices[i][j].item())
            print('best_scores[i][j]: ', best_scores[i][j].item())
            if best_indices[i][j].item() not in idx_score:
                idx_score[best_indices[i][j].item()] = [best_scores[i][j]]
                idx_model[best_indices[i][j].item()] = [loss_esbs[0][j]]
            else:
                idx_score[best_indices[i][j].item()].append(best_scores[i][j])
                idx_model[best_indices[i][j].item()].append(loss_esbs[0][j])
        topk_dic.append(idx_score)
    print('idx_score: ', idx_score)
    print('idx_model: ', idx_model)
    
    new_idx_score = defaultdict(torch.tensor)
    for key, value in idx_score.items():
            if len(value) == 1 and value[0].item()== -0:
                print('?')
                new_idx_score[key] = value[0]
            else:    
                value = torch.sum(torch.tensor(value))
                new_idx_score[key] = value
    # print('ㅣㅣ', sorted(new_idx_score.items(), key=lambda x: x [1], reverse=True))
    # print('ㅣ슬라이싱ㅣ', sorted(new_idx_score.items(), key=lambda x: x [1], reverse=True)[:beam_size])
    topk_result = dict(sorted(new_idx_score.items(), key=lambda x: x [1], reverse=True)[:beam_size])
    print('topk_result: ', topk_result)
    
    idx_loss = defaultdict(torch.tensor)
    for key, value in topk_result.items():
        loss_sum = torch.sum(torch.tensor(idx_model[key]))
        idx_loss[key] = loss_sum

    settled_topk = list(topk_result.keys())
    settled_scores = torch.div(torch.tensor(list(topk_result.values())), torch.tensor(list(idx_loss.values())))
    print('settled_scores: ', settled_scores)
    
    return settled_topk, settled_scores

def get_settled_topk_confi(best_scores, best_indices, beam_size = 4):
    """
    Group 별 confidence 고려 후 합이 가장 큰 단어 선택 -> settled output 결정
    """
    # top k 점수, 인덱스 가져오기
    best_indices = torch.transpose(best_indices, 0, 1)
    best_scores = torch.transpose(best_scores, 0, 1)
    
    vals = []   # vocab 번호
    counts = [] # vocab 빈도 수
    max_idx = [] # 최빈 vocab 인덱스
    topk_dic_confi = []   # topk의 idx-score confidence 합
    topk_result = [] # topk 최종 1등
    for i in range(beam_size):
        val, count = torch.unique(best_indices[i], return_counts = True)
        vals.append(val)
        counts.append(count)
        max_idx.append(torch.where(count == count.max()))
        
        # idx-scores confidence 합
        idx_confi = {}
        for j in range(len(best_indices[i])):
            if best_indices[i][j].item() not in idx_confi:
                idx_confi[best_indices[i][j].item()] = best_scores[i][j]
            else:
                # idx_confi[best_indices[i][j].item()].append(best_scores[i][j])
                idx_confi[best_indices[i][j].item()] = torch.add(idx_confi[best_indices[i][j].item()], best_scores[i][j])
        topk_dic_confi.append(idx_confi)
    
    
    # topk_dic_confi 에서 높은 confidence 가진 인덱스 확인
    for i in range(beam_size):
        # print('======')
        # print(min(topk_dic_confi[i],key=topk_dic_confi[i].get))
        topk_result.append(min(topk_dic_confi[i],key=topk_dic_confi[i].get))
    return topk_result


# python decoder_esb_softvoting.py --translate --data_dir ./wmt32k_data --model_dir ./outputs --eval_dir ./deu-eng

# dropout은 다른 gpu(id)에서 학습 -> torch.load에서 map_location 설정.
# python decoder_esb_softvoting.py --translate --data_dir ./wmt32k_data --model_dir ./outputs_dropout --eval_dir ./deu-eng

# dropout & alpha
# nohup python decoder_esb_softvoting_weigthed_loss_two.py --translate --data_dir ./wmt32k_data --model_dir ./outputs_dropout --eval_dir ./deu-eng --alpha_esb &

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
    
    alpha_esb = [0.0, 0.5]  # model 4, model 12, model 10

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
    model_path = models_dir + '/output_' + str(12) + '/last/models'
    model = utils.load_checkpoint2(model_path, device)
    models.append(model)
    
    
    pads = torch.tensor([trg_data['pad_idx']] * beam_size, device=device)
    pads = pads.unsqueeze(-1)

    # We'll find a target sequence by beam search.
    scores_history = [torch.zeros((beam_size,), dtype=torch.float,
                                  device=device)]
    

    eos_idx = trg_data['field'].vocab.stoi[trg_data['field'].eos_token]


    # f = open(f'{args.eval_dir}/testset_small.txt', 'r')
    f = open(f'{args.eval_dir}/oneline2.txt', 'r')
    dataset = f.readlines()
    f.close()
    
    
    f = open('./evaluation/esb_weighted_loss/hpys.txt', 'w')
    for data in tqdm(dataset):
        # Declare variables for each models
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
        settled_output = []
        
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
                best_scores = torch.tensor((), device = device)
                best_indices = torch.tensor((), device = device)
                
                # 각 모델 encoding 시작
                for i in range(m):
                    if idx > start_idx:
                        # print('targets[i]: ', targets[i])
                        targets[i] = torch.cat((targets[i], pads), dim=1)
                        # print('targets[i]: ', targets[i])
                        
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

                    
                    if args.alpha_esb:
                        length_penalties[i] = pow(((5. + idx + 1.) / 6.), alpha_esb[i])
                        scores[i] = scores[i] / length_penalties[i]
                        scores[i] = scores[i].view(-1)
                        
                    else:
                        length_penalty = pow(((5. + idx + 1.) / 6.), args.alpha)
                        scores[i] = scores[i] / length_penalty
                        scores[i] = scores[i].view(-1)
                
                    
                # Ensemble: Soft Voting
                for i in range(len(models)):    
                    
                    best_score, best_indice = scores[i].topk(beam_size, 0)
                    print('best_indice 모델 별: ', best_indice)
                    indices_history[i].append(best_indice)
                    
                    
                    best_indices = torch.cat((best_indices, best_indice.unsqueeze(0).float()), 0)
                    print('cat 하면서 Best+indices: ', best_indices)
                    best_scores = torch.cat((best_scores, best_score.unsqueeze(0)), 0)

                # Settled Output 결정
                loss_esbs = torch.tensor(loss_esb, device = device).repeat(beam_size, 1)
                settled_topk, settled_scores = get_settled_topk_with_index(loss_esbs, best_scores, best_indices)
                settled_topk = torch.tensor(settled_topk, device = device).long()
                settled_scores = torch.tensor(settled_scores, device = device)

                settled_output.append(settled_topk)
                
                #  tensor([-0.3423, -3.2212, -3.2346, -4.7794], device='cuda:0')
                # tensor([-0.3423, -3.2212, -3.2346, -4.7794])
                for i in range(len(models)):
                    scores_history[i].append(settled_scores)

                # Stop searching when the best output of beam is EOS.
                if settled_topk[0].item() % vocab_size == eos_idx:
                    break

                # 각 모델 다음 target update
                for i in range(len(models)):
                    targets[i] = update_targets(targets[i], settled_topk, idx, vocab_size)
                    

        # 모든 모델 같은 출력을 내기 때문에 0번째 모델의 결과를 출력
        result = get_result_sentence(settled_output, trg_data, vocab_size)
        f.write("Source: {}|Result: {}|Target: {}\n".format(source, result, target))
        
    f.close()

if __name__ == '__main__':
    main()
