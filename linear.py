import torch.cuda
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 目标数据是线性回归 因此单线性层即可完成训练
class MyLinearNN(nn.Module):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            self.linear = nn.Linear(1,1,True).cuda()
        else:
            self.linear = nn.Linear(1, 1, True)

    def forward(self,inp):
        return self.linear(inp)


# 数据加载
x=[]
y=[]
with open('../datasets/liner.csv','r') as f:
    lines = f.read().split('\n')

    x = [float(i.split(',')[0]) for i in lines]
    y = [float(i.split(',')[1]) for i in lines]



myNN = MyLinearNN()

# 损失函数 这里采用 MSELoss
loss_fn = nn.MSELoss() # 采取默认参数
if torch.cuda.is_available():
   loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.0001 # 试出来的
optimizer = torch.optim.SGD(myNN.parameters(),lr=learning_rate)

writer = SummaryWriter('../logs')
epochs = 2000 # 训练轮数
for i in range(epochs):
    for j in range(len(x)):
        x_tensor = torch.tensor(x[j], dtype=torch.float32).unsqueeze(dim=0) # unsqueeze 保证矩阵形状适用于计算
        y_tensor = torch.tensor(y[j], dtype=torch.float32).unsqueeze(dim=0)
        if torch.cuda.is_available():
            x_tensor = x_tensor.cuda()
            y_tensor = y_tensor.cuda()

        out = myNN(x_tensor) # 计算当前模型的输出
        loss = loss_fn(out,y_tensor) # 计算损失
        # 优化前清零梯度
        optimizer.zero_grad()
        loss.backward() # 损失值反向传播
        optimizer.step() # 执行一次优化
        if j == len(x)-1: # 每轮执行完毕进行一次打印
            print(myNN.state_dict())
            print('Loss:%f ,epoch: %d' % (loss.item(), i))
            writer.add_scalar('Linear Loss', loss.item(), i)



writer.close()





