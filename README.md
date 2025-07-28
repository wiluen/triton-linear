implement of triton kernel to replace F.linear()
torch.nn.modules.linear.py中forward替换为`triton_linear_3d(input, self.weight, self.bias)`
