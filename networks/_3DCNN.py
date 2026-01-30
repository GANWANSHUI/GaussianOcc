import math
import torch.nn.functional as F
import torch.nn as nn
import torch

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=None, activate_fun = None):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        self.noliner = nn.Sequential(activate_fun)

    def forward(self, x):
        return self.noliner(self.bn(self.conv(x)))

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out



def convbn_3d(in_planes, out_planes, kernel_size, stride, pad, conv_3d_types="3D"):

    if conv_3d_types == "3D":

        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
            nn.BatchNorm3d(out_planes))

    elif conv_3d_types == "P3D":  # 3*3*3　to 1*3*3 + 3*1*1

        return nn.Sequential(

            nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_planes))


class hourglass_PSMNet(nn.Module):
    def __init__(self, inplanes, conv_3d_types1, activate_fun):
        super(hourglass_PSMNet, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1, conv_3d_types = conv_3d_types1),
                                   activate_fun)

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, conv_3d_types = conv_3d_types1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1, conv_3d_types = conv_3d_types1),
                                   activate_fun)

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, conv_3d_types = conv_3d_types1),
                                   activate_fun)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(inplanes))  # +x

        self.activate_fun = nn.Sequential(activate_fun)

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = self.activate_fun(pre + postsqu)
        else:
            pre = self.activate_fun(pre)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = self.activate_fun(self.conv5(out) + presqu)  # in:1/16 out:1/8
        else:
            post = self.activate_fun(self.conv5(out) + pre)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post



class S3DCNN_challenge(nn.Module):
    def __init__(self,  input_planes = 64, out_planes = 1, planes = 16,  conv_3d_types1 = "3D", activate_fun = nn.ReLU(inplace=True), upsample = True, opt = None):
        super(S3DCNN_challenge, self).__init__()

        self.out_planes = out_planes

        self.upsample = upsample

        # self.opt = opt
        self.dres0 = nn.Sequential(convbn_3d(input_planes, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                     activate_fun,
                                     convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                     activate_fun)

        self.dres1 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types = conv_3d_types1),
                                   activate_fun,
                                   convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types = conv_3d_types1))

        self.dres2 = hourglass_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1, activate_fun = activate_fun)

        self.dres3 = hourglass_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1, activate_fun = activate_fun)

        self.dres4 = hourglass_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1, activate_fun = activate_fun)


        if self.upsample:

            self.upsample_net = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                        nn.ConvTranspose3d(planes * 2, planes * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                        nn.BatchNorm3d(planes * 2), activate_fun, convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1)) 


        self.classif3 = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                    activate_fun,

                                    convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                    activate_fun,

                                    convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                    activate_fun,

                                    convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                    activate_fun,

                                    convbn_3d(planes * 2, self.out_planes, 3, 1, 1, conv_3d_types=conv_3d_types1))
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def geo_param(self):
        return list(self.dres0.parameters()) + \
               list(self.dres1.parameters()) + \
               list(self.dres2.parameters()) + \
               list(self.dres3.parameters()) + \
               list(self.dres4.parameters()) + \
               list(self.classif1.parameters()) + \
               list(self.classif2.parameters()) + \
               list(self.classif3.parameters())
    

    def forward(self, cost):

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)

        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        
        if self.upsample:
            out3 = self.upsample_net(out3)

        cost3 = self.classif3(out3)
        
        return cost3





class S3DCNN(nn.Module):
    
    def __init__(self,  input_planes = 64, out_planes = 1, planes = 16,  conv_3d_types1 = "3D", activate_fun = nn.ReLU(inplace=True), opt = None):
        super(S3DCNN, self).__init__()
        self.out_planes = out_planes

        self.opt = opt
        self.dres0 = nn.Sequential(convbn_3d(input_planes, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                     activate_fun,
                                     convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                     activate_fun)

        self.dres1 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types = conv_3d_types1),
                                   activate_fun,
                                   convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types = conv_3d_types1))

        self.dres2 = hourglass_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1, activate_fun = activate_fun)

        self.dres3 = hourglass_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1, activate_fun = activate_fun)

        self.dres4 = hourglass_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1, activate_fun = activate_fun)


        # self.classif1 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
        #                               activate_fun,
        #                               nn.Conv3d(planes*2, out_planes, kernel_size=3, padding=1, stride=1,bias=False))

        # self.classif2 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
        #                               activate_fun,
        #                               nn.Conv3d(planes*2, out_planes, kernel_size=3, padding=1, stride=1,bias=False))

        if self.opt.upsample:

            self.upsample = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                        nn.ConvTranspose3d(planes * 2, planes * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                        nn.BatchNorm3d(planes * 2), activate_fun, convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1)) 


        self.classif3 = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                    activate_fun,

                                    convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                    activate_fun,

                                    convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                    activate_fun,

                                    convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                    activate_fun,

                                    convbn_3d(planes * 2, self.out_planes, 3, 1, 1, conv_3d_types=conv_3d_types1))
        

        if self.opt.use_semantic:

            self.classif_semantic = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      activate_fun,

                                      convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      activate_fun,

                                      convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      activate_fun,

                                      convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      activate_fun,

                                      convbn_3d(planes * 2, self.opt.semantic_classes, 3, 1, 1, conv_3d_types=conv_3d_types1),)
            
        if self.opt.flow != 'No':

            self.classif_flow = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      activate_fun,

                                      convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      activate_fun,

                                      convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      activate_fun,

                                    #   convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                    #   activate_fun,

                                      convbn_3d(planes * 2, 3, 3, 1, 1, conv_3d_types=conv_3d_types1),)
                        
            # self.opacity_activation = torch.sigmoid
            self.flow_activation = torch.nn.functional.tanh 


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if self.opt.render_type == 'density':
            pass
    

    def geo_param(self):
        return list(self.dres0.parameters()) + \
               list(self.dres1.parameters()) + \
               list(self.dres2.parameters()) + \
               list(self.dres3.parameters()) + \
               list(self.dres4.parameters()) + \
               list(self.classif1.parameters()) + \
               list(self.classif2.parameters()) + \
               list(self.classif3.parameters())
    
    def sem_head_param(self):
        if self.opt.use_semantic:
            return self.classif_semantic.parameters()
        else:
            return None

    def forward(self, cost):

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)

        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)


        if self.opt.upsample:

            out3 = self.upsample(out3)


        if self.opt.use_semantic:

            if self.opt.last_free:
                out = self.classif_semantic(out3)

            else:
                semantic = self.classif_semantic(out3)
                cost3 = self.classif3(out3)
                out = torch.cat([semantic, cost3], dim=1)

            return [out]

        elif self.opt.flow != 'No':

            flow_3d = self.classif_flow(out3)
            # flow_3d = self.flow_activation(flow_3d) 

            cost3 = self.classif3(out3)

            out = torch.cat([cost3, flow_3d], dim=1)

            return [out]


        else:

            cost3 = self.classif3(out3)
                 
            return [cost3]





def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape
    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    elif len(shp) == 5:
        pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    return F.pad(this, pad)




if __name__ == '__main__':

    pass

