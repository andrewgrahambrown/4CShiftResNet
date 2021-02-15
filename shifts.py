import torch
import torch.nn as nn

class shift_4C(nn.Module):
    #Move inputs right,left,down,up (+ unmoved)
    def __init__(self):
        super(shift_4C, self).__init__()

    def forward(self,x):
        xs = x.shape
        xs5= xs[1]//5 #just 3x3 kernels for now

        if (xs5==0):
            #Could be that we could not run this at all as the input size is too small
            print("Channel number too small to implement shift")
            raise KeyboardInterrupt

        split_sections = []
        for i in range(4):
            split_sections.append(xs5)
        split_sections.append(xs[1]-xs5*4)
        xsplit = torch.split(x,split_size_or_sections=split_sections,dim=1)
        xsplitN = []
        #0
        r = xsplit[0].roll([ 1, 0],[2,3])
        r[:,:,0,:] = 0
        xsplitN.append(r)
        #1
        r = xsplit[1].roll([ 0,-1],[2,3])
        r[:,:,:,xs[3]-1] = 0
        xsplitN.append(r)
        #2
        r = xsplit[2].roll([-1, 0],[2,3])
        r[:,:,xs[2]-1,:] = 0
        xsplitN.append(r)
        #3
        r = xsplit[3].roll([ 0, 1],[2,3])
        r[:,:,:,0] = 0
        xsplitN.append(r)
        #4 This is the unmoved portion
        xsplitN.append(xsplit[4])
        retx = torch.cat(xsplitN,dim=1)
        return retx


class shift_8C(nn.Module):
    #Move inputs right,left,down,up and diagonals (+ unmoved)
    def __init__(self):
        super(shift_8C, self).__init__()

    def forward(self,x):        
        xs = x.shape
        xs9= xs[1]//9

        split_sections = []
        for i in range(8):
            split_sections.append(xs9)
        split_sections.append(xs[1]-xs9*8)
        xsplit = torch.split(x,split_size_or_sections=split_sections,dim=1)

        xsplitN = []
        #Now go through each of the 8 positions and roll the tensors
        #0
        r = xsplit[0].roll([ 1, 1],[2,3])
        r[:,:,0,:] = 0
        r[:,:,:,0] = 0
        xsplitN.append(r)
        #1
        r = xsplit[1].roll([ 1, 0],[2,3])
        r[:,:,0,:] = 0
        xsplitN.append(r)
        #2
        r = xsplit[2].roll([ 1,-1],[2,3])
        r[:,:,0,:] = 0
        r[:,:,:,xs[3]-1] = 0
        xsplitN.append(r)
        #3
        r = xsplit[3].roll([ 0,-1],[2,3])
        r[:,:,:,xs[3]-1] = 0
        xsplitN.append(r)
        #4
        r = xsplit[4].roll([-1,-1],[2,3])
        r[:,:,xs[2]-1,:] = 0
        r[:,:,:,xs[3]-1] = 0
        xsplitN.append(r)
        #5
        r = xsplit[5].roll([-1, 0],[2,3])
        r[:,:,xs[2]-1,:] = 0
        xsplitN.append(r)
        #6
        r = xsplit[6].roll([-1, 1],[2,3])
        r[:,:,xs[2]-1,:] = 0
        r[:,:,:,0] = 0
        xsplitN.append(r)
        #7
        r = xsplit[7].roll([ 0, 1],[2,3])
        r[:,:,:,0] = 0
        xsplitN.append(r)
        #8 This is the unmoved portion
        xsplitN.append(xsplit[8])
        retx = torch.cat(xsplitN,dim=1)
        return retx
