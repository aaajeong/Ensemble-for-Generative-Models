single = open('./valid/single/model4/all_bleu.txt', 'r')
esb = open('./valid/esb/1:0/all_bleu.txt', 'r')

single_bleus = single.readline()
esb_bleus = esb.readline()

single_bleus = single_bleus.split(',')
esb_bleus = esb_bleus.split(',')
count = 0

for i in range(100):
    if single_bleus[i] != esb_bleus[i]:
        print(i)
        count += 1
print('count: ', count)
