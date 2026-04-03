import csv
import datetime

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment


# data = pd.read_csv('weather.csv',encoding = 'gb18030')
# print(data)

# X = data.iloc[:,:6]
# X = pd.DataFrame(X)
# y = data.iloc[:,6]
# pca = PCA(n_components=2)
# new_pca = pd.DataFrame(pca.fit_transform(X))
# print(new_pca)
# #主成分系数
# print(pd.DataFrame(pca.components_,columns=X.columns,index=['PC1','PC2']))

# kms = KMeans(n_clusters=6,n_init='auto')
# Y = kms.fit_predict(X)
# data['class'] = Y
# data.to_csv("weather_new_test.csv",index=False)

# plt.scatter(new_pca[0],new_pca[1],c=Y)
# plt.show()

# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# # 真实标签
# y_true = data.iloc[:, 6].values - 1

# # KMeans标签
# y_kmeans = Y

# # ===== 1. ARI =====
# ari = adjusted_rand_score(y_true, y_kmeans)
# print(f"ARI: {ari}")

# # ===== 2. NMI =====
# nmi = normalized_mutual_info_score(y_true, y_kmeans)
# print(f"NMI: {nmi}")

# # ===== 3. 混淆矩阵 =====
# cm = confusion_matrix(y_true, y_kmeans)

# # 取负值（因为是最大匹配）
# row_ind, col_ind = linear_sum_assignment(-cm)

# # 建立映射关系
# mapping = {col: row for row, col in zip(row_ind, col_ind)}

# # 重映射KMeans标签
# y_kmeans_aligned = np.array([mapping[i] for i in y_kmeans])

# # 新混淆矩阵
# cm_new = confusion_matrix(y_true, y_kmeans_aligned)

# plt.figure(figsize=(6,5))
# sns.heatmap(cm_new, annot=True, fmt='d', cmap='Blues')
# plt.xlabel("KMeans(predicted)")
# plt.ylabel("FCM (true)")
# plt.title("Confusion Matrix")
# plt.gca().invert_yaxis()
# plt.show()

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
        with open('weather_new_test.csv','r') as f:
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
        with open('weather_new_test.csv','r') as f:
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
    torch.save(net,'model_mlp.pth')

    #加载模型
    net = torch.load('model_mlp.pth', weights_only=False)
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

        
            

