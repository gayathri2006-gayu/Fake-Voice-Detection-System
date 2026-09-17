import torch
import torch.nn as nn

print("Torch version:", torch.__version__)

class CNN(nn.Module):
    def _init_(self):
        print("Before super init")
        super()._init_()
        print("After super init")
        print("Type of self._parameters:", type(self._parameters) if hasattr(self, '_parameters') else "NO ATTR")
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        print("After setting conv1")
        print("Has conv1 in dict?", 'conv1' in self._dict_)
        print("Has conv1 in _modules?", 'conv1' in self._modules if hasattr(self, '_modules') else "NO _modules")

model = CNN()