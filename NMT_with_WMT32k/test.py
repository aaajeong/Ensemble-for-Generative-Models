import torch
# dictlist = [
#     {'height' : 170, 'weight': 60, 'name':'홍길동'},
#     {'height' : 160, 'weight': 90, 'name':'이몽룡'},
#     {'height' : 165, 'weight': 55, 'name':'성춘향'},
#     {'height' : 180, 'weight': 70, 'name':'대조영'},
#     {'height' : 180, 'weight': 85, 'name':'김개똥'},
#     {'height' : 165, 'weight': 65, 'name':'아무개'}
    
# ]
 
# # height 높은 순, weight 낮은 순
# sorted_dict = sorted(dictlist, key = lambda x : (-x['height'], x['weight']))
# sorted_dict

new_idx_score = {4744.0: torch.tensor(0.7574), 78.0: torch.tensor(0.3275), 3120.0: torch.tensor(0), 12.0: torch.tensor(0.1564), 43723.0: torch.tensor(0), 117015.0: torch.tensor(0.), 11322.0: torch.tensor(0)}

idx_model = {4744.0: [torch.tensor(1), torch.tensor(0)], 78.0: [torch.tensor(1)], 3120.0: [torch.tensor(0)], 12.0: [torch.tensor(1)], 43723.0: [torch.tensor(0)], 117015.0: [torch.tensor(1)], 11322.0: [torch.tensor(0)]}

score_loss = []
for key, value in new_idx_score.items():
    item = {}
    item['index'] = key
    item['score'] = value
    item['loss'] = torch.sum(torch.tensor(idx_model[key]))
    score_loss.append(item)

# score_loss = [
#     {'index' : 4744.0, 'score' : 0.7574, 'loss': [1,0]},
#     {'index' : 78.0, 'score' : 0.3275, 'loss': [1]},
#     {'index' : 3120, 'score' : 0, 'loss': [0]},
#     {'index' : 12.0, 'score' : 0.1564, 'loss': [1]},
#     {'index' : 43723.0, 'score' : 0, 'loss': [0]},
#     {'index' : 117015.0, 'score' : 0, 'loss': [1]},
#     {'index' : 11322.0, 'score' : 0, 'loss': [0]}
# ]
# score_loss = [
#     {'index' : 4744.0, 'score' : 0.7574, 'loss': 1},
#     {'index' : 78.0, 'score' : 0.3275, 'loss': 1},
#     {'index' : 3120, 'score' : 0, 'loss': 0},
#     {'index' : 12.0, 'score' : 0.1564, 'loss': 1},
#     {'index' : 43723.0, 'score' : 0, 'loss': 0},
#     {'index' : 117015.0, 'score' : 0, 'loss': 1},
#     {'index' : 11322.0, 'score' : 0, 'loss': 0}
# ]

# code로 구현
# [{'index': 4744.0, 'score': tensor(0.7574), 'loss': tensor(1)}, 
#  {'index': 78.0, 'score': tensor(0.3275), 'loss': tensor(1)}, 
#  {'index': 3120.0, 'score': tensor(0), 'loss': tensor(0)}, 
#  {'index': 12.0, 'score': tensor(0.1564), 'loss': tensor(1)}, 
#  {'index': 43723.0, 'score': tensor(0), 'loss': tensor(0)}, 
#  {'index': 117015.0, 'score': tensor(0.), 'loss': tensor(1)}, 
#  {'index': 11322.0, 'score': tensor(0), 'loss': tensor(0)}]

sorted_score_loss = sorted(score_loss, key = lambda x : (-x['score'], -x['loss']))[:4]

settled_topk = [i['index'] for i in sorted_score_loss]
settled_scores = torch.tensor([i['score'] for i in sorted_score_loss])
setteld_loss = torch.tensor([[i['loss'] for i in sorted_score_loss]])

settled_scores = torch.div(settled_scores, setteld_loss)    # sum/weighted_loss 값

idx_score_origin = {4744.0: [torch.tensor(-3.8906), torch.tensor(-0.)], 78.0: [torch.tensor(-4.3206)], 3120.0: [torch.tensor(-0.)], 12.0: [torch.tensor(-4.4917)], 43723.0: [torch.tensor(-0.)], 117015.0: [torch.tensor(-4.6481)], 11322.0: [torch.tensor(-0.)]}

settled_scores_origin = [torch.sum(torch.tensor(idx_score_origin[i['index']])) for i in sorted_score_loss]

print(settled_scores)
print(settled_scores_origin)

# [{'index': 4744.0, 'score': tensor(0.7574), 'loss': tensor(1)}, 
#  {'index': 78.0, 'score': tensor(0.3275), 'loss': tensor(1)}, 
#  {'index': 12.0, 'score': tensor(0.1564), 'loss': tensor(1)}, 
#  {'index': 117015.0, 'score': tensor(0.), 'loss': tensor(1)}, 
#  {'index': 3120.0, 'score': tensor(0), 'loss': tensor(0)}, 
#  {'index': 43723.0, 'score': tensor(0), 'loss': tensor(0)}, 
#  {'index': 11322.0, 'score': tensor(0), 'loss': tensor(0)}]