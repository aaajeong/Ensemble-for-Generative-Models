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

# python decoder_esb_ckp.py --translate --data_dir ./wmt32k_data --model_dir ./output_ckp/last/models --eval_dir ./deu-eng
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--eval_dir', type=str, required=False)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--translate', action='store_true')
    
    args = parser.parse_args()

    beam_size = args.beam_size

    # Load fields.
    if args.translate:
        src_data = torch.load(args.data_dir + '/source.pt')
    trg_data = torch.load(args.data_dir + '/target.pt')

    # Load a saved model.
    device = torch.device('cpu' if args.no_cuda else 'cuda:1')
    # model = utils.load_checkpoint(args.model_dir, device)
    # model2 = utils.load_checkpoint(args.model_dir, device)
    model = utils.load_checkpoint_ckp(190, args.model_dir, device, is_eval=False)
    model2 = utils.load_checkpoint_ckp(195, args.model_dir, device, is_eval=False)
    model3 = utils.load_checkpoint(args.model_dir, device)

    pads = torch.tensor([trg_data['pad_idx']] * beam_size, device=device)
    pads = pads.unsqueeze(-1)

    # We'll find a target sequence by beam search.
    scores_history = [torch.zeros((beam_size,), dtype=torch.float,
                                  device=device)]
    scores_history2 = [torch.zeros((beam_size,), dtype=torch.float,
                                  device=device)]
    scores_history3 = [torch.zeros((beam_size,), dtype=torch.float,
                                  device=device)]
    esb_scores_history = [torch.zeros((beam_size,), dtype=torch.float,
                                  device=device)]
    # indices_history = []
    # cache = {}

    eos_idx = trg_data['field'].vocab.stoi[trg_data['field'].eos_token]


    f = open(f'{args.eval_dir}/testset_small.txt', 'r')
    # f = open(f'{args.eval_dir}/oneline.txt', 'r')
    dataset = f.readlines()
    f.close()
    
    # f = open('./evaluation/single/hpys_m10.txt', 'w')
    f = open('./evaluation/esb/ckp/hpys2.txt', 'w')
    for data in tqdm(dataset):
        cache = {}
        cache2 = {}
        cache3 = {}
        
        indices_history = []
        indices_history2 = []
        indices_history3 = []
        
        esb_indices_history = []
        
        
        target, source = data.strip().split('\t')   # 원래 데이터셋 형태: en -> de
        
        if args.translate:
            # sentence = input('Source? ')
            sentence = source   # de

        # Encoding inputs.
        if args.translate:
            start_time = time.time()
            enc_output, src_mask = encode_inputs(sentence, model, src_data,
                                                beam_size, device)
            enc_output2, src_mask2 = encode_inputs(sentence, model2, src_data,
                                                beam_size, device)
            enc_output3, src_mask3 = encode_inputs(sentence, model3, src_data,
                                                beam_size, device)
            targets = pads
            targets2 = pads
            targets3 = pads
            start_idx = 0
        else:
            enc_output, src_mask = None, None
            sentence = input('Target? ').split()
            for idx, _ in enumerate(sentence):
                sentence[idx] = trg_data['field'].vocab.stoi[sentence[idx]]
            sentence.append(trg_data['pad_idx'])
            targets = torch.tensor([sentence], device=device)
            targets2 = torch.tensor([sentence], device=device)
            targets3 = torch.tensor([sentence], device=device)
            start_idx = targets.size(1) - 1
            start_time = time.time()

        with torch.no_grad():
            for idx in range(start_idx, args.max_length):
                if idx > start_idx:
                    targets = torch.cat((targets, pads), dim=1)
                    targets2 = torch.cat((targets2, pads), dim=1)
                    targets3 = torch.cat((targets3, pads), dim=1)
                t_self_mask = utils.create_trg_self_mask(targets.size()[1],
                                                        device=targets.device)
                t_self_mask2 = utils.create_trg_self_mask(targets2.size()[1],
                                                        device=targets2.device)
                t_self_mask3 = utils.create_trg_self_mask(targets3.size()[1],
                                                        device=targets3.device)

                t_mask = utils.create_pad_mask(targets, trg_data['pad_idx'])
                t_mask2 = utils.create_pad_mask(targets2, trg_data['pad_idx'])
                t_mask3 = utils.create_pad_mask(targets3, trg_data['pad_idx'])
                
                pred = model.decode(targets, enc_output, src_mask,
                                    t_self_mask, t_mask, cache)
                
                pred2 = model2.decode(targets2, enc_output2, src_mask2,
                                    t_self_mask2, t_mask2, cache2)
                pred3 = model3.decode(targets3, enc_output3, src_mask3,
                                    t_self_mask3, t_mask3, cache3)
                
                pred = pred[:, idx].squeeze(1)
                pred2 = pred2[:, idx].squeeze(1)
                pred3 = pred3[:, idx].squeeze(1)
                vocab_size = pred.size(1)

                pred = F.log_softmax(pred, dim=1)
                pred2 = F.log_softmax(pred2, dim=1)
                pred3 = F.log_softmax(pred3, dim=1)
                
                if idx == start_idx:
                    scores = pred[0]
                    scores2 = pred2[0]
                    scores3 = pred3[0]
                else:
                    scores = scores_history[-1].unsqueeze(1) + pred
                    scores2 = scores_history2[-1].unsqueeze(1) + pred2
                    scores3 = scores_history3[-1].unsqueeze(1) + pred3
                    
                length_penalty = pow(((5. + idx + 1.) / 6.), args.alpha)
                scores = scores / length_penalty
                scores = scores.view(-1)
                
                scores2 = scores2 / length_penalty
                scores2 = scores2.view(-1)
                
                scores3 = scores3 / length_penalty
                scores3 = scores3.view(-1)

                # Ensemble: Checkpoints
                # ensemble_scores = torch.zeros_like(scores)
                ensemble_scores = scores + scores2 + scores3
                esb_scores, esb_indices = ensemble_scores.topk(beam_size, 0)
                esb_scores_history.append(esb_scores)
                esb_indices_history.append(esb_indices)
                
                best_scores, best_indices = scores.topk(beam_size, 0)
                best_scores2, best_indices2 = scores2.topk(beam_size, 0)
                best_scores3, best_indices3 = scores3.topk(beam_size, 0)
                scores_history.append(best_scores)
                indices_history.append(best_indices)
                
                scores_history2.append(best_scores2)
                indices_history2.append(best_indices2)
                
                scores_history3.append(best_scores3)
                indices_history3.append(best_indices3)

                # Stop searching when the best output of beam is EOS.
                if best_indices[0].item() % vocab_size == eos_idx:
                    break

                targets = update_targets(targets, best_indices, idx, vocab_size)
                targets2 = update_targets(targets2, best_indices2, idx, vocab_size)
                targets3 = update_targets(targets3, best_indices3, idx, vocab_size)

        result = get_result_sentence(indices_history, trg_data, vocab_size)
        result2 = get_result_sentence(indices_history2, trg_data, vocab_size)
        result3 = get_result_sentence(indices_history, trg_data, vocab_size)
        esb_result = get_result_sentence(esb_indices_history, trg_data, vocab_size)
        
        # print(result)
        # print(result2)
        # print(esb_result)
        f.write("Source: {}|Result: {}|Target: {}\n".format(source, esb_result, target))
        

        # print("Elapsed Time: {:.2f} sec".format(time.time() - start_time))
        
    f.close()

if __name__ == '__main__':
    main()
