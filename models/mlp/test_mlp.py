import csv
import datetime
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from sklearn.preprocessing import StandardScaler



class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,6),
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
        self.start = datetime.datetime.now()
    
    def forward(self,out):
        out = self.fc(out)
        return out
    
    def train_step(self,x,y):
        out = self.forward(x)#前向传播
        loss = self.loss_function(out,y)#计算损失
        self.optimizer.zero_grad()#清空梯度
        loss.backward()#反向传播
        self.optimizer.step()#更新参数
        return loss


    def test(self,x):
        out = self.forward(x)
        return out
    
    def get_data(self):
        with open('./data/weather_new_test.csv','r') as f:
            data = csv.reader(f)
            data = [row for row in data]
            data = data[1:1500]#标题行不能算
        inputs = []
        labels = []

        for row in data:
            one_hot = [0 for i in range(6)]
            one_hot[int(row[6]) - 1] = 1
            labels.append(one_hot)

            input = row[:6]
            input = [float(i) for i in input]
            inputs.append(input)

        inputs = torch.tensor(inputs).float()
        labels = torch.tensor(labels).float()

        return inputs,labels
        
    def get_test_data(self):
        with open('./data/weather_new_test.csv','r') as f:
            data = csv.reader(f)
            data = [row for row in data]
            data = data[1500:]
        inputs = []
        labels = []
        for row in data:
            input = row[:6]
            label = [row[6]]
            input = [float(i) for i in input]
            label = [float(i) for i in label]
            inputs.append(input)
            labels.append(label)
        
        inputs = torch.tensor(inputs).float()
        labels = torch.tensor(labels).float()
        
        return inputs,labels
            

if __name__ == '__main__':

    Epoches = 100
    Batch_size = 50

    net = MyNet()
    x_data_train,y_data_train = net.get_data()
    scaler = StandardScaler()
    x_data_train = scaler.fit_transform(x_data_train)
    x_data_train = torch.tensor(x_data_train).float()
    my_dataset = Data.TensorDataset(x_data_train,y_data_train)
    my_dataloader = Data.DataLoader(
        dataset=my_dataset,
        batch_size=Batch_size,
        shuffle=True,
    )
    for epoch in range(Epoches):
        epoch_start_time = datetime.datetime.now()
        for step,(batch_x,batch_y) in enumerate(my_dataloader):
            loss = net.train_step(batch_x,batch_y)
        epoch_end_time = datetime.datetime.now()
        print(f"epoch:{epoch},loss:{loss.item()},time:{epoch_end_time - epoch_start_time}")

    train_end_time = datetime.datetime.now()
    print(f"训练完成，耗时{train_end_time - net.start}")
    
    #保存模型
    torch.save(net,'./models/model_mlp.pth')

    #加载模型
    net = torch.load('./models/model_mlp.pth', weights_only=False)
    x_data_test,y_data_test = net.get_test_data()
    x_data_test = scaler.transform(x_data_test)
    x_data_test = torch.tensor(x_data_test).float()
    my_dataset = Data.TensorDataset(x_data_test,y_data_test)
    my_dataloader = Data.DataLoader(
        dataset=my_dataset,
        batch_size=Batch_size,
        shuffle=False,
    )
    num_success = 0
    num_total = 317
    test_start_time = datetime.datetime.now()
    for step,(batch_x,batch_y) in enumerate(my_dataloader):
        out = net.test(batch_x)
        # print(step)
        for index , i in enumerate(out):
            i = i.detach().numpy()
            j = np.argmax(i)
            # print(f"输出为{j+1}标签为{batch_y[index][0].item()}")
            loss = j+1 - batch_y[index][0].item()
            if loss == 0:
                num_success += 1
    test_end_time = datetime.datetime.now()
    print(f"准确率为{num_success/num_total}, 耗时{test_end_time - test_start_time}")

        
            

