import torch
import random
import torchvision
import numpy as np
import torch.nn.functional as func

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

def generator(n, m, min_val, max_val, func, error=1):
    X = []
    y = []
    diff = max_val - min_val

    for i in range(n):
        x0 = [round(random.random() * diff + min_val, 2) for j in range(m)]
        X.append(x0)
        b = int(func(*x0))

        y.append(b if random.random() < error else 1-b)


    return np.array(X), np.array(y).reshape((-1,1))



generator(100, 2, 0, 5, lambda x, y: x*x)
arr_x = range(10)
x = torch.Tensor([[i] for i in arr_x])

y = torch.Tensor([[f(i)] for i in arr_x])

model = MyModule()

loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x)
    l = loss(y_pred, y)
    print(epoch, l.item())
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

print("w:", model.linear.weight.item())
print("b:", model.linear.bias.item())