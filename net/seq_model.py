import torch.nn as nn
import torch.nn.functional as F
import torch
from .utils import *
from .resnet import resnet50
from .conv_lstm import ConvLSTMCell


class SeqModel(nn.Module):
    def __init__(self, nf, in_chan, device):
        super(SeqModel, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.device=device
        self.encoder=resnet50(input_channels=3,no_top=True, multi_outputs=True)

        # conv_channels = [64, 256, 512, 1024, 2048]
        deconv_channels = [512, 512, 256, 128, 64, 32]

        self.decode1=deconv(deconv_channels[0],deconv_channels[1])
        self.decode2=deconv(deconv_channels[1],deconv_channels[2])
        self.decode3=deconv(deconv_channels[2]+1,deconv_channels[3])
        self.decode4=deconv(deconv_channels[3]+1,deconv_channels[4])
        self.decode5=deconv(deconv_channels[4]+1,deconv_channels[5])

        self.output2 = conv(deconv_channels[2], 1, k=1, stride=1, padding=0,output=True)
        self.output3 = conv(deconv_channels[3], 1, k=1, stride=1, padding=0,output=True)
        self.output4 = conv(deconv_channels[4], 1, k=1, stride=1, padding=0,output=True)
        self.output5 = conv(deconv_channels[5], 1, k=1, stride=1, padding=0,output=True)

        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        # self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
        #                                        hidden_dim=nf,
        #                                        kernel_size=(3, 3),
        #                                        bias=True)

        # self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
        #                                        hidden_dim=nf,
        #                                        kernel_size=(3, 3),
        #                                        bias=True)

        # self.decoder_CNN = nn.Conv3d(in_channels=nf,
        #                              out_channels=1,
        #                              kernel_size=(1, 3, 3),
        #                              padding=(0, 1, 1))

        self.conv_block = nn.Sequential(
            conv(512, 512,stride=1,activation='relu'),
            conv(512, 256,stride=1,activation='relu'),
            conv(256, 6,stride=1,output=True),
        )

    def forward(self, x):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        pose_init=torch.Tensor(b,6)
        pose_init.to(self.device)
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        # h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        # h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        feature_seq = self.autoencoder(x, seq_len, h_t, c_t, h_t2, c_t2)

        depth_maps, poses=[], [pose_init]
        for feature in feature_seq:
            depth_maps.append(self.decode(feature))
        for feature in feature_seq[1:]:
            poses.append(self.pose_decode(feature))

        return depth_maps, poses

    def autoencoder(self, x, seq_len, h_t, c_t, h_t2, c_t2):

        output_tensor = []
        
        # encoder
        for t in range(seq_len):
            input_tensor=self.encoder(x[:,t,...])
            h_t, c_t = self.encoder_1_convlstm(
                input_tensor=input_tensor,
                cur_state=[h_t, c_t]
            )  # we could concat to provide skip conn here

            h_t2, c_t2 = self.encoder_2_convlstm(
                input_tensor=h_t,
                cur_state=[h_t2, c_t2]
            )  # we could concat to provide skip conn here
            output_tensor.append(h_t2)

        # decoder
        # for t in range(future_step):
        #     h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
        #                                          cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
        #     h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
        #                                          cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
        #     encoder_vector = h_t4
        #     outputs += [h_t4]  # predictions
        return output_tensor

    def decode(self, x):

        d = self.decode2(self.decode1(x))  # 1024->512, 28x28
        d1 = F.sigmoid(self.output2(d)) # 1x28x28

        d = self.decode3(cat([d, d1]))  # 256x56x56
        d2 = F.sigmoid(self.output3(d))

        d = self.decode4(cat([d, d2]))  # 128x112x112
        d3 = F.sigmoid(self.output4(d))

        d = self.decode5(cat([d, d3]))  # 64x224x224
        d4 = F.sigmoid(self.output5(d))

        return [d4, d3, d2, d1] if self.training else d4
        

    def decode_pose(self, x):
        x=self.conv_block(x).mean(3).mean(2)
        return 0.01*x.view(-1,6)