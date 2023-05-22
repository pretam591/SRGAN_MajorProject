class RefineNet(nn.Module):
    def __init__(self, num_features=16):
        # B,3,96,96
        super(RefineNet, self).__init__()
        self.kernels = HierConv(3,num_features,5)
        self.weights_conv_r = nn.Sequential(nn.Conv2d(3,1,5,2,2),nn.ReLU())
        self.weights_linear_r = nn.Linear(48*48,16)
        self.weights_conv_g = nn.Sequential(nn.Conv2d(3,1,5,2,2),nn.ReLU())
        self.weights_linear_g = nn.Linear(48*48,16)
        self.weights_conv_b = nn.Sequential(nn.Conv2d(3,1,5,2,2),nn.ReLU())
        self.weights_linear_b = nn.Linear(48*48,16)
    def forward(self, x):
        features = torch.tanh(self.kernels(x))
        x_flat_r = torch.flatten(self.weights_conv_r(x),1)
        weights_r = torch.sigmoid(self.weights_linear_r(x_flat_r))
        x_flat_g = torch.flatten(self.weights_conv_g(x),1)
        weights_g = torch.sigmoid(self.weights_linear_g(x_flat_g))
        x_flat_b = torch.flatten(self.weights_conv_b(x),1)
        weights_b = torch.sigmoid(self.weights_linear_b(x_flat_b))

        delta_r = (weights_r.unsqueeze(-1).unsqueeze(-1)*features).sum(dim=1)
        delta_g = (weights_g.unsqueeze(-1).unsqueeze(-1)*features).sum(dim=1)
        delta_b = (weights_b.unsqueeze(-1).unsqueeze(-1)*features).sum(dim=1)
        
        delta_rgb = torch.stack([delta_r,delta_g,delta_b],dim = 1)
        out = x + delta_rgb
        return out  