import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ControlModule(nn.Module):
    def __init__(self,num_patches=197,num_features=768,num_controls=64, config=None,layer_id=None):
        super().__init__()
        self.num_patches = num_patches
        self.num_features = num_features
        self.num_controls = num_controls

        self.down_proj = nn.Linear(self.num_features, self.num_controls)
        self.up_proj = nn.Linear(self.num_controls, self.num_features)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)



    def forward(self, x):   
        down = self.down_proj(x)        
        up = self.up_proj(down) 
        affinity_matrix = torch.bmm(up,up.permute(0,2,1))
        affinity_matrix = F.softmax(affinity_matrix,dim=-1)       
        up = torch.bmm(affinity_matrix,up) + up
        return up