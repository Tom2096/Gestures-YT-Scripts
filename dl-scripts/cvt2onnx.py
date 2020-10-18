#!coding=utf-8
import torch

from eval import Net

if __name__=='__main__':
    
    checkpoint = 'e:/projects/gestures-app/dl-scripts/checkpoint/net_v0.pth'
    buf = torch.load(checkpoint)

    net = Net()
    
    net.load_state_dict(buf)
    net.eval()

    dummy_input = torch.zeros(1, 21, 2)
    torch.onnx.export(net, dummy_input, 'e:/projects/gestures-app/front/public/model.onnx')

