from utils import *
from line import Line
from tqdm import trange
import torch
import torch.optim as optim
# Define dimension, batch_size, learning_rate, epochs, etc. for LINE model
dimension = 5
graph_path = "../LINE_HOPE/data.edgelist"
batch_size = 10
learning_rate = 0.25
order = 2
epochs = 15
neg_sample_size = 6
model_save_path = "embeddingvector.txt"

# Create dict of distribution of a graph data
edges_dict, nodes_dict, weights, node_degrees, max_index = makeDist(graph_path)

edges_alias_sampler = VoseAlias(edges_dict)
nodes_alias_sampler = VoseAlias(nodes_dict)

batchrange = int(len(edges_dict) / batch_size)
line = Line(max_index + 1, dimension=dimension, order=order)

opt = optim.SGD(line.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lossdata = {"it": [], "loss": []}
it = 0

print("\nTraining data...")
for epoch in range(epochs):
    print("Epoch - {}".format(epoch))
    for b in trange(batchrange):
        sampled_edges = edges_alias_sampler.sample_n(batch_size)
        batch = list(makeData(sampled_edges, neg_sample_size, weights, node_degrees,
                              nodes_alias_sampler))

        batch = torch.LongTensor(batch)
        v_i = batch[:, 0]
        v_j = batch[:, 1]
        negative_samples = batch[:, 2:]
        line.zero_grad()
        loss = line(v_i, v_j, negative_samples, device)
        loss.backward()
        opt.step()
        lossdata["loss"].append(loss.item())
        lossdata["it"].append(it)
        it += 1
print("DONE")
torch.save(line, "{}".format(model_save_path))