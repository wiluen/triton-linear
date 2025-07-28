implement of triton kernel to replace F.linear()

torch.nn.modules.linear.py `forward` to `triton_linear_3d(input, self.weight, self.bias)`
