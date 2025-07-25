import torch
import random

def syn_data(w,b,num_examples):
    x = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(x,w) + b
    y += torch.normal(0,0.1,y.shape)
    return x,y.reshape(-1,1)
features,labels = syn_data(torch.tensor([2,-3.4]),-2.3,1000)
# print(features,labels)
# w = [2,-3.4] ; b = -2.3

w = torch.tensor([0.,0],requires_grad=True)
b = torch.zeros(1,requires_grad=True)
# initial : w = [0.0] ; b = 0

def data_iter(batch_size, features, labels):
    '''
    接收批量大小、特征矩阵和标签向量作为输入，生成大小为 batch_size的小批量。
    每个小批量包含一组特征和标签。
    #'''
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def S_loss(y_dat,y):
    return (y_dat - y.reshape(y_dat.shape)) ** 2 / 2

def sgd(params,lr,num_patch):
    for param in params:
        with torch.no_grad():
            param -= lr * param.grad / num_patch
            param.grad.zero_()

def linreg(x,w,b):
    return torch.matmul(x,w) + b

lr = 0.01
num_epochs = 10
batch_size = 20

for epoch in range(num_epochs):
    for f,l in data_iter(batch_size, features, labels):
        y = linreg(f,w,b)
        loss = S_loss(y,l).sum()
        loss.backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        print(f'epoch{epoch+1} : w = {w.tolist()} , b = {float(b)} , loss = {loss}')


# 成功运行！！！！！
# epoch1 : w = [0.7417816519737244, -1.3636187314987183] , b = -0.88954758644104 , loss = 65.6395492553711
# epoch2 : w = [1.2082706689834595, -2.1803760528564453] , b = -1.435402750968933 , loss = 55.47071075439453
# epoch3 : w = [1.502877950668335, -2.6687676906585693] , b = -1.7705614566802979 , loss = 11.273307800292969
# epoch4 : w = [1.6878941059112549, -2.960826873779297] , b = -1.9762831926345825 , loss = 5.173161506652832
# epoch5 : w = [1.8043017387390137, -3.135877847671509] , b = -2.1019155979156494 , loss = 1.4500066041946411
# epoch6 : w = [1.8773295879364014, -3.240992307662964] , b = -2.178678512573242 , loss = 0.8594446778297424
# epoch7 : w = [1.9233933687210083, -3.3039379119873047] , b = -2.226024866104126 , loss = 0.3019145131111145
# epoch8 : w = [1.9520491361618042, -3.3416290283203125] , b = -2.2553138732910156 , loss = 0.1333329975605011
# epoch9 : w = [1.9701696634292603, -3.364044427871704] , b = -2.273017644882202 , loss = 0.09657508879899979
# epoch10 : w = [1.9815664291381836, -3.3779349327087402] , b = -2.283602237701416 , loss = 0.081427663564682
