import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import gamma


# 设定随机种子，确保结果可复现
torch.manual_seed(1234)
np.random.seed(1234)

mu = 6
sigma = 0.5




class Gausin(nn.Module):
    def __init__(self):
        super(Gausin, self).__init__()
    def forward(self, x):
        return -torch.sin(x) * torch.exp((-x * x) / 2)


# 神经网络定义
class NN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.act = Gausin()


    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


# compute Norm
def ND(x):
    y = norm.pdf(x, loc=mu, scale=sigma)
    return y



x_train = np.linspace(0, 10, 100)


# y_train = ND(x_train)
# y_train = ND(x_train) + ND(x_train+2)
# y_train = ND(x_train) + ND(x_train+2) +ND(x_train+4)
y_train = ND(x_train) + ND(x_train+2) +10*ND(x_train+4)



x_train_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

x_test = torch.tensor(np.linspace(0, 10, 1000), dtype=torch.float32).view(-1, 1)

# y_test = ND(x_test)
# y_test = ND(x_test) + ND(x_test+2)
# y_test = ND(x_test) + ND(x_test+2)  + ND(x_test+4)
y_test = ND(x_test) + ND(x_test+2)  + 10*ND(x_test+4)

model = NN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练神经网络
epochs = 8000
with open("prob_rl2_4.100_gaus.txt", "w") as log_file:
    log_file.write("l2\n")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(x_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            y_pred_test = model(x_test).detach().numpy()
            rl2 = np.sqrt(np.sum((y_pred_test - y_test) ** 2) / np.sum(y_test ** 2))
            log_file.write(f"{rl2}\n")
            print(f"Epoch [{epoch}/{epochs}], l2: {rl2:6f}")

# 训练完毕后，绘制结果
model.eval()

y_pred_test = model(x_test).detach().numpy()


plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, label='True PDF', linestyle='--',color='black')
plt.plot(x_test, y_pred_test, label='NN Approximation', color='blue')
plt.legend()
plt.title('Approximation')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.show()

