from torch import nn
import torch
import numpy as np
import random


class LinearRegressionModel(nn.Module):
    def __init__(self, dim_in, dim_out, seed=100):
        super(LinearRegressionModel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.linear(x)


class MultiClassLogisticRegressionModel(nn.Module):
    def __init__(self, num_classes=10, dim=784, seed=100):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.linear(x)  # No softmax here!


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=2, seed=100):
        super().__init__()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
# class SimpleClassifierMultilayer(nn.Module):
#     def __init__(self, input_dim, output_dim=2, seed=100):
#         super().__init__()
#         if seed is not None:
#             random.seed(seed)
#             np.random.seed(seed)
#             torch.manual_seed(seed)
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False
#         self.linear = nn.Linear(input_dim, output_dim)
#         self.linear = nn.Linear(output_dim, output_dim)

#     def forward(self, x):
#         return self.linear(x)
