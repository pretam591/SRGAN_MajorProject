import torch.nn.functional as F
from torch.nn import Module

class HierConv(Module):    # HIERARCHICAL CONVOLUTIONS
    def __init__(self,IC,OC,level=5,remap_K=1,remap_S=1,remap_P=0):
        super(HierConv, self).__init__()
        self.level = level
        assert(self.level>=5 and self.level%2==1), "level should be a odd number greater than or equal to 5"
        self.layers = {}
        for l in range(5,self.level+1,2):
            self.layers[str(l)]=nn.ModuleList([nn.Conv2d(IC,OC,k,1,k//2).cuda() for k in range(3,l+1,2)])
        self.output_dim=0
        for i in range(5, self.level+1, 2):
            for j in range (1, i-1, 2):
                self.output_dim += j**2 
        self.remap = nn.Conv2d(OC*self.output_dim,OC,remap_K,remap_S,remap_P,1,OC)
    def forward(self, x):
        outs={}
        for l in range(5,self.level+1,2):
            outs[str(l)]=[self.layers[str(l)][i](x) for i in range(l//2)]
        B,C,H,W = outs['5'][0].shape
        final_out=torch.zeros(B,C*self.output_dim,H,W).cuda()
        for b in range(B):
            for c in range(C):
                stackedFeatures = []
                for l in range(5,self.level+1,2):
                    d=-1
                    stackedFeatures.append(outs[str(l)][d][b,c,:,:])
                    d-=1
                    for p in range(3,l,2):
                        for sh in range(0,p):                                                                                
                            for sw in range(0,p):
                                stackedFeatures.append(F.pad(outs[str(l)][d][b,c,:,:],(sh,p-1-sh,sw,p-1-sw))[p//2:-(p//2),p//2:-(p//2)]) 
                        d-=1
                m = self.output_dim 
                final_out[b,m*c:m*c+m,:,:]=torch.stack(stackedFeatures, 0)
        output = self.remap(final_out)
        return(output)

hc = HierConv(3,8,5).cuda()
input = torch.randn(4,3,256,256).cuda() #b,c,h,w
output = hc(input)
output.shape 