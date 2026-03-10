import torch
from piq_net import PIQNet

nodes = 50
features = 8
timesteps = 10

node_features = torch.rand(nodes,features)
edge_index = torch.randint(0,nodes,(2,200))
temporal_sequence = torch.rand(1,timesteps,64)

model = PIQNet(node_features=features)

piq = model(node_features,edge_index,temporal_sequence)

print("Planetary IQ:",piq.item())
