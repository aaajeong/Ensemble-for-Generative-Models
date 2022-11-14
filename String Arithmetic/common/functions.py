# coding: utf-8
from common.np import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# 앙상블-서바이벌 모델 : 모델 결과에서 가장 많이 등장한 답 출력
def get_semi_answer(id_list, max_list):
    vals,counts = np.unique(id_list, return_counts=True)
    occurences = np.where(counts == counts.max())

    # id-score 딕셔너리 생성
    id_score = {}
    for i in range(len(id_list)):
        if id_list[i] not in id_score:
            id_score[id_list[i]] = [max_list[i]]
        else:
            id_score[id_list[i]].append(max_list[i])

    # 세미정답 결정
    scores = []
    if len(occurences[0]) != 1:
        for i in range(len(occurences[0])):
            scores.append(np.mean(np.array(id_score[vals[occurences[0][i]]])))
    
        scores = np.array(scores)

        result = vals[occurences[0][np.argmax(scores)]]
        return result
    else:
        result = vals[occurences[0][0]]
        return result