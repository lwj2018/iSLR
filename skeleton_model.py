import torch
import torch.nn as nn
from torch.nn import functional as F

class skeleton_model(nn.Module):

    def __init__(self, num_class, in_channel=2,
                            length=32,num_joint=10):
        # T N D
        super(skeleton_model, self).__init__()
        self.num_class = num_class
        self.in_channel = in_channel
        self.length = length
        self.num_joint = num_joint

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,64,1,1,padding=0),
            nn.ReLU()
            )
        self.conv2 = nn.Conv2d(64,32,(3,1),1,padding=(1,0))
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.num_joint,32,3,1,padding=1),
            nn.MaxPool2d(2)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32,64,3,1,padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.convm1 = nn.Sequential(
            nn.Conv2d(in_channel,64,1,1,padding=0),
            nn.ReLU()
            )
        self.convm2 = nn.Conv2d(64,32,(3,1),1,padding=(1,0))
        self.convm3 = nn.Sequential(
            nn.Conv2d(self.num_joint,32,3,1,padding=1),
            nn.MaxPool2d(2)
            )
        self.convm4 = nn.Sequential(
            nn.Conv2d(32,64,3,1,padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
                
        self.conv5 = nn.Sequential(
            nn.Conv2d(128,128,3,1,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128,256,3,1,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7 = nn.Sequential(
            nn.Linear(256*(length//16)*(32//16),256),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256,self.num_class)

    def forward(self, input):
        '''
            input: N D T V
        '''
        input = input.permute(0,3,1,2)
        N, D, T, V = input.size()
        motion = input[:,:,1::,:]-input[:,:,0:-1,:]
        motion = F.upsample(motion,size=(T,V),mode='bilinear').contiguous()

        out = self.conv1(input)
        out = self.conv2(out)
        out = out.permute(0,3,2,1).contiguous()
        out = self.conv3(out)
        out = self.conv4(out)

        outm = self.convm1(motion)
        outm = self.convm2(outm)
        outm = outm.permute(0,3,2,1).contiguous()
        outm = self.convm3(outm)
        outm = self.convm4(outm)

        out = torch.cat((out,outm),dim=1)
        out = self.conv5(out)
        out = self.conv6(out)

        out = out.view(out.size(0),-1)
        out = self.fc7(out)
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out


        