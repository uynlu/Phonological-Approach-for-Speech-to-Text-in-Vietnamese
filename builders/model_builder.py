import torch
from .registry import Registry

META_MODEL = Registry(name="MODEL")

def build_model(config, vocab):
    model = META_MODEL.get(config.model)(config, vocab)
    model = model.to(torch.device(config.device))
    
    return model
