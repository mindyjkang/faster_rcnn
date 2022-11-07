from torch import nn


class RPN(nn.Module):

    def __init__(self, in_channels, anchor_num):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, 2*anchor_num, kernel_size=1, stride=1)
        self.reg_layer = nn.Conv2d(in_channels, 4*anchor_num, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.conv(input)
        cls_logit = self.cls_layer(x)
        cls_score = self.softmax(cls_logit)
        reg_output = self.reg_layer(x)
        return cls_score, reg_output
    