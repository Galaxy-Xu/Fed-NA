from torch import nn

# class MLP(nn.Module):
#     def __init__(self, num_classes):
#         super(MLP, self).__init__()
#         self.layer_input = nn.Linear(784, 256)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()
#         self.layer_hidden1 = nn.Linear(256, 64)
#         self.layer_hidden2 = nn.Linear(64, num_classes)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#
#         x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
#         x = self.layer_input(x)
#         x = self.dropout(x)
#         x = self.relu(x)
#         # 通过第二个隐藏层
#         x = self.layer_hidden1(x)
#         x = self.dropout(x)  # 如果你想在第二个隐藏层后也使用dropout
#         x = self.relu(x)
#         x = self.layer_hidden2(x)
#
#         return self.softmax(x)



from torch import nn

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(3072, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 64)
        self.layer_hidden2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        # 通过第二个隐藏层
        x = self.layer_hidden1(x)
        x = self.dropout(x)  # 如果你想在第二个隐藏层后也使用dropout
        x = self.relu(x)
        x = self.layer_hidden2(x)

        return self.softmax(x)



