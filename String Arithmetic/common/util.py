# coding: utf-8
import sys
sys.path.append('..')
import os
from common.np import *
from common.functions import get_semi_answer


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def cos_similarity(x, y, eps=1e-8):
    '''코사인 유사도 산출

    :param x: 벡터
    :param y: 벡터
    :param eps: '0으로 나누기'를 방지하기 위한 작은 값
    :return:
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''유사 단어 검색

    :param query: 쿼리(텍스트)
    :param word_to_id: 단어에서 단어 ID로 변환하는 딕셔너리
    :param id_to_word: 단어 ID에서 단어로 변환하는 딕셔너리
    :param word_matrix: 단어 벡터를 정리한 행렬. 각 행에 해당 단어 벡터가 저장되어 있다고 가정한다.
    :param top: 상위 몇 개까지 출력할 지 지정
    '''
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def convert_one_hot(corpus, vocab_size):
    '''원핫 표현으로 변환

    :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
    :param vocab_size: 어휘 수
    :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def create_co_matrix(corpus, vocab_size, window_size=1):
    '''동시발생 행렬 생성

    :param corpus: 말뭉치(단어 ID 목록)
    :param vocab_size: 어휘 수
    :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
    :return: 동시발생 행렬
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def ppmi(C, verbose=False, eps = 1e-8):
    '''PPMI(점별 상호정보량) 생성

    :param C: 동시발생 행렬
    :param verbose: 진행 상황을 출력할지 여부
    :return:
    '''
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('%.1f%% 완료' % (100*cnt/total))
    return M


def create_contexts_target(corpus, window_size=1):
    '''맥락과 타깃 생성

    :param corpus: 말뭉치(단어 ID 목록)
    :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
    :return:
    '''
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print('퍼플렉서티 평가 중 ...')
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush()

    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl


def eval_seq2seq(model, question, correct, id_to_char,
                 verbos=False, is_reverse=False):
    correct = correct.flatten()
    # 머릿글자
    start_id = correct[0]
    correct = correct[1:]

    guess = model.generate(question, start_id, len(correct))

    # 문자열로 변환
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    guess = ''.join([id_to_char[int(c)] for c in guess])

    if verbos:
        if is_reverse:
            question = question[::-1]

        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
        print('Q', question)
        print('T', correct)

        is_windows = os.name == 'nt'

        if correct == guess:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
            print(mark + ' ' + guess)
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' ' + guess)
        print('---')

    return 1 if guess == correct else 0

# 모델 앙상블 (소프트 보팅)
def eval_seq2seq_esb(model_list, question, correct, id_to_char,
                 verbos=False, is_reverse=False):

    correct = correct.flatten()
    # 머릿글자
    start_id = correct[0]
    correct = correct[1:]

    model_list = model_list

    for model in model_list:
        # h : hidden state?
        h = model.encoder.forward(question)
        model.decoder.lstm.set_state(h)

    sample_size = len(correct)
    voting_result = []
    sample_id = start_id
    
    # 이건 이중 for 돌릴때
    result1 = []
    result2 = []
    model_num = len(model_list)
    id_list = [[] * model_num for i in range(model_num)]
    # print(id_list)

    for _ in range(sample_size):
        x_list = []
        out_list = []
        out_list2 = []
        score_list = []
        # input x
        for i in range(len(model_list)):
            x = np.array(sample_id).reshape(1, 1)
            x_list.append(x)

        # 디코더의 forward (x)
        for i in range(len(model_list)):
            out = model_list[i].decoder.embed.forward(x_list[i])
            out_list.append(out)

        # 디코더의 lstm
        for i in range(len(model_list)):
            out = model_list[i].decoder.lstm.forward(out_list[i])
            out_list2.append(out)

        # 디코더의 affine
        for i in range(len(model_list)):
            score = model_list[i].decoder.affine.forward(out_list2[i])
            score_list.append(score)

        # argmax id
        for i in range(len(model_list)):
            score_id = np.argmax(score_list[i].flatten())
            id_list.append(score_id)
        # print('id_list: ', id_list)

        # Soft Voting
        # score_sum = np.empty(()) --> 나중에 수정
        # for i in range(len(model_list)):
        #     score_sum += score_list[i]
        score_sum = score_list[0] + score_list[1] + score_list[2] + score_list[3] + score_list[4]
        # print('score_sum : ', score_sum)

        voting_id = np.argmax(score_sum.flatten())
        # print('voting_id: ', voting_id)

        voting_result.append(voting_id)
        # print('voting_id가 추가된 voring_result: ', voting_result)

        sample_id = voting_id

    voting_guess = voting_result


    # 문자열로 변환
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    voting_guess = ''.join([id_to_char[int(c)] for c in voting_guess])

    # results = []
    # results.append(voting_guess)
    # return voting_guess

    if verbos:
        if is_reverse:
            question = question[::-1]

        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
        print('Q', question)
        print('T', correct)

        is_windows = os.name == 'nt'

        if correct == voting_guess:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
            print(mark + ' ' + voting_guess)
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' ' + voting_guess)
        print('---')

    return 1 if voting_guess == correct else 0

# 모델 앙상블 - 서바이벌
def eval_seq2seq_survival(model_list, question, correct, id_to_char, verbos=False, is_reverse=False):
    correct = correct.flatten()
    # 머릿글자
    start_id = correct[0]
    correct = correct[1:]

    model_list = model_list
    suvi_models = model_list.copy()

    for model in suvi_models:
        # h : hidden state?
        h = model.encoder.forward(question)
        model.decoder.lstm.set_state(h)

    sample_size = len(correct)
    survival_result = []
    sample_id = start_id

    model_num = len(model_list)
    # id_list = [[] * model_num for i in range(model_num)]

    # correct의 길이만큼 반복(prediction)
    for _ in range(sample_size):
#         print('===============', _, '번째 자리===============')
        # id_list = [[] * model_num for i in range(model_num)]
        id_list = []
        x_list = []
        out_list = []
        out_list2 = []
        score_list = []
        max_list = []
        
        if model_num > 2:
            # input x
            for i in range(model_num):
                x = np.array(sample_id).reshape(1, 1)
                x_list.append(x)
            
            # 디코더의 forward (x)
            for i in range(model_num):
                out = suvi_models[i].decoder.embed.forward(x_list[i])
                out_list.append(out)
            
            # 디코더의 lstm
            for i in range(model_num):
                out = suvi_models[i].decoder.lstm.forward(out_list[i])
                out_list2.append(out)

            # 디코더의 affine
            for i in range(model_num):
                score = suvi_models[i].decoder.affine.forward(out_list2[i])
                score_list.append(score)

            # argmax id, max value
            for i in range(model_num):
                score_id = np.argmax(score_list[i].flatten())
                id_list.append(score_id)
                max_list.append(score_list[i][0][0][score_id])
        
            # ========== Survival Ensemble =========== 
            # 1. 모델에서 가장 많이 등장한 인덱스 번호 찾기 (다 다르게 나올 경우 제일 첫번째 값으로 됨)
            # semi_answer = max(id_list, key=id_list.count)
            semi_answer = get_semi_answer(id_list, max_list)

            # 2. 세미정답을 출력한 살아남은 모델 --> winner
            #    세미정답을 출력하지 않은 실패한 모델 --> loser
            winner = list(np.where(np.array(id_list) == semi_answer)[0])
            loser = list(np.where(np.array(id_list) != semi_answer)[0]) 

            # 3. winner 모델만 다시 서바이벌 참여할 수 있도록 참여모델 업데이트
            update_sm = [] # suvi_models 업데이트

            for win in winner:
                update_sm.append(suvi_models[win])

            # 업데이트한 모델을 다시 서바이벌 모델 리스트에 대입
            suvi_models = update_sm
            
#             print('참가한 모델의 예측: ', id_list)
#             print('다수결 결과: ', semi_answer)
#             print('살아남은 모델: ', winner)
#             print('탈락한 모델: ', loser)
#             print('-------------------------------------------')

            # 4. 다음 서바이벌 경쟁을 위해 탈락한 모델(loser) 개수 제외
            model_num -= len(loser)

            # 5. semi_answer 에 해당하는 결과 저장
            survival_result.append(semi_answer)

            # 6. 다음 id 넘겨주기
            sample_id = semi_answer
        elif model_num == 2:
            # 살아있는 모델 수가 2개
            # input x
            for i in range(model_num):
                x = np.array(sample_id).reshape(1, 1)
                x_list.append(x)
            
            # 디코더의 forward (x)
            for i in range(model_num):
                out = suvi_models[i].decoder.embed.forward(x_list[i])
                out_list.append(out)
            
            # 디코더의 lstm
            for i in range(model_num):
                out = suvi_models[i].decoder.lstm.forward(out_list[i])
                out_list2.append(out)

            # 디코더의 affine
            for i in range(model_num):
                score = suvi_models[i].decoder.affine.forward(out_list2[i])
                score_list.append(score)

            # argmax id, max value
            for i in range(model_num):
                score_id = np.argmax(score_list[i].flatten())
                id_list.append(score_id)
                max_list.append(score_list[i][0][0][score_id])
            
            # ========== Survival Ensemble =========== 
            if id_list[0] == id_list[1]:
                semi_answer = id_list[0]
                survival_result.append(semi_answer)
#                 print('참가한 모델의 예측: ', id_list)
#                 print('다수결 결과: ', semi_answer)
#                 print('-------------------------------------------')
                sample_id = semi_answer
            else:
                semi_answer = id_list[max_list.index(max(max_list))]
                survival_result.append(semi_answer)
#                 print('참가한 모델의 예측: ', id_list)
#                 print('다수결 결과: ', semi_answer)
#                 print('-------------------------------------------')

                # 이제 모델 1개로 진행(2개가 다른 답이 나왔으니깐)
                loser = list(np.where(np.array(id_list) != semi_answer)[0])
                for l in loser:
                    del suvi_models[l]
                model_num -= 1

                #print('탈락한 모델(2개중): ', loser)
                sample_id = semi_answer
        else:
            # 살아남은 모델 1개
            # input x
            x = np.array(sample_id).reshape(1, 1)
            x_list.append(x)
            
            # 디코더의 forward (x)
            out = suvi_models[0].decoder.embed.forward(x_list[0])
            out_list.append(out)
            
            # 디코더의 lstm
            out = suvi_models[0].decoder.lstm.forward(out_list[0])
            out_list2.append(out)

            # 디코더의 affine
            score = suvi_models[0].decoder.affine.forward(out_list2[0])
            score_list.append(score)

            # argmax id, max value
            score_id = np.argmax(score_list[0].flatten())
            survival_result.append(score_id)
#             print('최후 1개 모델의 예측: ', score_id)
#             print('-------------------------------------------')
            sample_id = score_id

    survival_guess = survival_result
#     print('서바이벌 결과: ',survival_guess)
    # 문자열로 변환
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    survival_guess = ''.join([id_to_char[int(c)] for c in survival_guess])
#     print('문자열로 변환한 결과: ',survival_guess)

    if verbos:
        if is_reverse:
            question = question[::-1]

        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
#         print('Q', question)
#         print('T', correct)

        is_windows = os.name == 'nt'

        if correct == survival_guess:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
#             print(mark + ' ' + survival_guess)
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
#             print(mark + ' ' + survival_guess)
#         print('---')

    return 1 if survival_guess == correct else 0

# 모델 앙상블 - 진짜 다수결
def eval_seq2seq_real(model_list, question, correct, id_to_char, verbos=False, is_reverse=False):
    correct = correct.flatten()
    # 머릿글자
    start_id = correct[0]
    correct = correct[1:]

    model_list = model_list
    suvi_models = model_list.copy()

    for model in suvi_models:
        # h : hidden state?
        h = model.encoder.forward(question)
        model.decoder.lstm.set_state(h)

    sample_size = len(correct)
    # sample_id = start_id
    sample_ids = []
    model_num = len(model_list)
    for i in range(model_num):
        sample_ids.append(start_id)
    candi_list = [[] * model_num for i in range(model_num)]
    affine_list = [[] * model_num for i in range(model_num)]

    # correct의 길이만큼 반복(prediction)
    for _ in range(sample_size):
        id_list = []
        x_list = []
        out_list = []
        out_list2 = []
        score_list = []
        max_list = []
        # input x
        for i in range(model_num):
            x = np.array(sample_ids[i]).reshape(1, 1)
            x_list.append(x)

        # 디코더의 forward (x)
        for i in range(model_num):
            out = model_list[i].decoder.embed.forward(x_list[i])
            out_list.append(out)

        # 디코더의 lstm
        for i in range(model_num):
            out = model_list[i].decoder.lstm.forward(out_list[i])
            out_list2.append(out)

        # 디코더의 affine
        for i in range(model_num):
            score = model_list[i].decoder.affine.forward(out_list2[i])
            score_list.append(score)

        # argmax id & max affine
        for i in range(model_num):
            score_id = np.argmax(score_list[i].flatten())
            id_list.append(score_id)
            affine = score_list[i][0][0][score_id]
            affine_list[i].append(affine)

        # 각 모델의 번역 결과 저장
        for i in range(model_num):
            candi_list[i].append(id_list[i])
        
        # 다음 단어 예측을 위한 id 넘겨주기
        for i in range(model_num):
            sample_ids[i] = id_list[i]
    
    # 문자열로 변환
    for i in range(len(candi_list)):
        ans = ''.join([id_to_char[int(c)] for c in candi_list[i]])
        candi_list[i] = ans
        # candi_list[i] = ''.join([id_to_char[int(c)] for c in candi_list[i]])

    # =========== 진짜 다수결 시작 ============
    # 각 모델 결과의 affine 평균 값
    affine_list = np.array(affine_list)
    affine_avg = []
    for m in affine_list:
        affine_avg.append(np.mean(m))

    # 각 모델 결과가 같은지/아닌지 그룹핑
    group = {}
    index = []
    for m in range(model_num):
        if candi_list[m] in group:
            group[candi_list[m]].append(m)
        else:
            index = [m]
            group[candi_list[m]] = index

    # 그룹 별 어파인 평균값의 합
    affine_sum = {}
    for idx in group.values():
        sum = 0
        for m in idx:
            sum += affine_avg[m]
        affine_sum[tuple(idx)] = sum

    # 가장 높은 어파인합 값을 가진 번역 결과 출력
    real_maj = max(affine_sum, key = affine_sum.get)

    for key, value in group.items():
        if value == list(real_maj):
            majority_guess = key

    # 문제/정답 문자열로 변환
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])

    if verbos:
        if is_reverse:
            question = question[::-1]

        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
        print('Q', question)
        print('T', correct)

        is_windows = os.name == 'nt'

        if correct == majority_guess:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
            print(mark + ' ' + majority_guess)
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' ' + majority_guess)
        print('---')

    return 1 if majority_guess == correct else 0

def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print('%s(을)를 찾을 수 없습니다.' % word)
            return

    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x
