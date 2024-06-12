# @Author:Yilu Wu
# @Time:2024/2/26
# @Description:

import torch.nn


class MLP(torch.nn.Module):
    def __init__(self, dim, T):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(6 * dim, 4 * dim)
        self.dp = torch.nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(4 * dim, 2*dim)
        self.linears = torch.nn.ModuleList()
        for i in range(T):
            self.linears.append(torch.nn.Linear(2*dim,dim))

    def forward(self, input):
        x = self.linear1(input)
        x = self.relu(x)
        x = self.dp(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dp(x)
        output = []
        for linear in self.linears:
            output.append(linear(x))
        return output
