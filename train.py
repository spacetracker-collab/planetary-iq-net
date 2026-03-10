import torch
import torch.nn as nn
from piq_net import PIQNet
from dataset import generate_dataset

model = PIQNet(node_features=8)

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

loss_fn = nn.MSELoss()

X_nodes,X_edges,X_time,Y = generate_dataset(200)

for epoch in range(50):

    total_loss = 0

    for x,e,t,y in zip(X_nodes,X_edges,X_time,Y):

        pred = model(x,e,t)

        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(epoch,total_loss)
