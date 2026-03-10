import torch


def generate_dataset(samples=100):

    nodes = 50
    features = 8
    timesteps = 10

    X_nodes = []
    X_edges = []
    X_time = []
    Y = []

    for _ in range(samples):

        node_features = torch.rand(nodes,features)

        edge_index = torch.randint(0,nodes,(2,200))

        temporal_sequence = torch.rand(1,timesteps,64)

        piq = torch.tensor([node_features.mean()*1000])

        X_nodes.append(node_features)
        X_edges.append(edge_index)
        X_time.append(temporal_sequence)
        Y.append(piq)

    return X_nodes,X_edges,X_time,Y
