from models.effnet import EffNet
from models.vgg16 import VGGNet
from models.vit import VIT
from models.swinv2 import SwinNet

def get_model(model_name:str, model_args:dict):
    if model_name == 'effnet':
        return EffNet(**model_args)
    elif model_name == 'vgg16':
        return VGGNet(**model_args)
    elif model_name == 'vit':
        return VIT(**model_args)
    elif model_name == 'swinv2':
        return SwinNet(**model_args)
    else: 
        raise ValueError(f'Model name {model_name} is not valid.')

if __name__ == '__main__':
    pass