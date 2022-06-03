from jinja2 import Template
import numpy as np
from train import MyModel
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Convert a simple autoencoder to CUDA C++ code.')
    parser.add_argument('-fuse_relu', type=int, default=1, help='Fuse ReLU with convolutions')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    fuse_relu = False if args.fuse_relu == 0 else True
    
    with open('weights.template', 'r') as f:
        template = Template(f.read())

    with open('main.template', 'r') as f:
        infer_template = Template(f.read())

    weight_list = []
    infer_list = []
    model = MyModel()
    model.eval()
    model.load_state_dict(torch.load('ckpt.pth', map_location=torch.device('cpu')))

    with torch.no_grad():
        for i, module in enumerate(model.children()):
            if isinstance(module, torch.nn.Conv2d):
                weight_list.append({
                    'name': f'conv_{i}',
                    'data': np.ascontiguousarray(np.transpose(module.weight.numpy(), [0, 2, 3, 1])),
                    'bias': np.ascontiguousarray(module.bias.numpy())
                })
                infer_list.append({
                    'name': f'conv_{i}',
                    'relu': 1 if (i < 10 and fuse_relu) else 0,
                    'type': 'conv',
                    'pad': 1 if i < 10 else 0
                })
            elif isinstance(module, torch.nn.ConvTranspose2d):
                weight_list.append({
                    'name': f'conv_{i}',
                    'data': np.ascontiguousarray(np.transpose(module.weight.numpy()[..., ::-1, ::-1], [1, 2, 3, 0])),
                    'bias': np.ascontiguousarray(module.bias.numpy())
                })
                infer_list.append({
                    'name': f'conv_{i}',
                    'relu': 1 if (i < 10 and fuse_relu) else 0,
                    'type': 'transposedconv'
                })
            elif isinstance(module, torch.nn.Sigmoid):
                infer_list.append({
                    'type': 'sigmoid'
                })
            elif isinstance(module, torch.nn.MaxPool2d):
                infer_list.append({
                    'type': 'maxpool'
                })
            elif isinstance(module, torch.nn.ReLU) and not fuse_relu:
                infer_list.append({
                    'type': 'relu'
                })

    data = {
        'weight_list': weight_list,
        'layer_list': infer_list
    }

    with open('weights.h', 'w') as f:
        f.write(template.render(data) + '\n')
    with open('main.cu', 'w') as f:
        f.write(infer_template.render(data) + '\n')
