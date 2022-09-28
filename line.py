import torch
import torch.nn as nn
import torch.nn.functional as F

class Line(nn.Module):
    def __init__(self, size, dimension=2, order=1):
        super(Line, self).__init__()

        self.dimension = dimension
        self.order = order
        self.nodes_embeddings = nn.Embedding(size, dimension)

        assert order in [1, 2], print("In LINE, we only have first-order and second-order!")

        # Create context nodes (neighbor nodes) embedding when applying second-order
        if order == 2:
            
            self.context_nodes_embed = nn.Embedding(size, dimension)
            # Initiate data for context
            self.context_nodes_embed.weight.data = self.context_nodes_embed.weight.data.uniform_(
                -.5, .5) / dimension

        # Initiate data for nodes embedding
        self.nodes_embeddings.weight.data = self.nodes_embeddings.weight.data.uniform_(
            -.5, .5) / dimension

    def forward(self, v_i, v_j, negsamples, device):

        v_i = self.nodes_embeddings(v_i).to(device)

        if self.order == 2:
            v_j = self.context_nodes_embed(v_j).to(device)
            negative_nodes = -self.context_nodes_embed(negsamples).to(device)
        else:
            v_j = self.nodes_embeddings(v_j).to(device)
            negative_nodes = -self.nodes_embeddings(negsamples).to(device)

        multiply_positive_batch = torch.mul(v_i, v_j)
        positive_batch = F.logsigmoid(torch.sum(multiply_positive_batch, dim=1))

        multiply_negative_batch = torch.mul(v_i.view(len(v_i), 1, self.dimension), negative_nodes)
        negative_batch = torch.sum(
            F.logsigmoid(
                torch.sum(multiply_negative_batch, dim=2)
            ),dim=1)
        loss = positive_batch + negative_batch
        return -torch.mean(loss)