import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def set_frozen_layers(self, name_list:list):
        # some parameters are directly loaded from pre-trained models, and they are typically frozen and not required to save
        if not isinstance(name_list, list):
            name_list = [name_list]
        self.frozen_layers = name_list

    def get_learnable_params(self):
        learnable_params = {}
        for param, value in self.state_dict().items():
            to_save = True
            for frozen_layer in self.frozen_layers:
                if frozen_layer in param.split('.'):
                    to_save = False  # not save

            if to_save:
                learnable_params[param] = value
        # print(learnable_params.keys())
        return learnable_params

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

