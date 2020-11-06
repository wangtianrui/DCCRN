# coding: utf-8
# Author：WangTianRui
# Date ：2020/11/3 16:49
# from base.BaseModel import *
import torch.nn as nn
import torch
from utils.conv_stft import *
from utils.complexnn import *


class DCCRN(nn.Module):
    def __init__(self,
                 rnn_layer=2, rnn_hidden=256,
                 win_len=400, hop_len=100, fft_len=512, win_type='hanning',
                 use_clstm=True, use_cbn=False, masking_mode='E',
                 kernel_size=5, kernel_num=(32, 64, 128, 256, 256, 256)
                 ):
        super(DCCRN, self).__init__()
        self.rnn_layer = rnn_layer
        self.rnn_hidden = rnn_hidden

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.win_type = win_type

        self.use_clstm = use_clstm
        self.use_cbn = use_cbn
        self.masking_mode = masking_mode

        self.kernel_size = kernel_size
        self.kernel_num = (2,) + kernel_num

        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, self.win_type, 'complex', fix=True)
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, self.win_type, 'complex', fix=True)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                        self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if self.use_clstm:
            rnns = []
            for idx in range(rnn_layer):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1] if idx == 0 else self.rnn_hidden,
                        hidden_size=self.rnn_hidden,
                        batch_first=False,
                        projection_dim=hidden_dim * self.kernel_num[-1] if idx == rnn_layer - 1 else None
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_hidden,
                num_layers=2,
                dropout=0.0,
                batch_first=False
            )
            self.transform = nn.Linear(self.rnn_hidden, hidden_dim * self.kernel_num[-1])
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            self.kernel_num[idx - 1]),
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        )
                    )
                )
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, x):
        stft = self.stft(x)
        # print("stft:", stft.size())
        real = stft[:, :self.fft_len // 2 + 1]
        imag = stft[:, self.fft_len // 2 + 1:]
        # print("real imag:", real.size(), imag.size())
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_phase = torch.atan2(imag, real)
        spec_complex = torch.stack([real, imag], dim=1)[:, :, 1:]  # B,2,256
        # print("spec", spec_mags.size(), spec_phase.size(), spec_complex.size())

        out = spec_complex
        encoder_out = []
        for idx, encoder in enumerate(self.encoder):
            out = encoder(out)
            # print("encoder out:", out.size())
            encoder_out.append(out)
        B, C, D, T = out.size()
        out = out.permute(3, 0, 1, 2)
        if self.use_clstm:
            r_rnn_in = out[:, :, :C // 2]
            i_rnn_in = out[:, :, C // 2:]
            r_rnn_in = torch.reshape(r_rnn_in, [T, B, C // 2 * D])
            i_rnn_in = torch.reshape(i_rnn_in, [T, B, C // 2 * D])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [T, B, C // 2, D])
            i_rnn_in = torch.reshape(i_rnn_in, [T, B, C // 2, D])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)
        else:
            out = torch.reshape(out, [T, B, C * D])
            out, _ = self.enhance(out)
            out = self.transform(out)
            out = torch.reshape(out, [T, B, C, D])
        out = out.permute(1, 2, 3, 0)
        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])
        if self.masking_mode == 'E':
            mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
            real_phase = mask_real / (mask_mags + 1e-8)
            imag_phase = mask_imag / (mask_mags + 1e-8)
            mask_phase = torch.atan2(
                imag_phase,
                real_phase
            )
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags * spec_mags
            est_phase = spec_phase + mask_phase
            real = est_mags * torch.cos(est_phase)
            imag = est_mags * torch.sin(est_phase)
        elif self.masking_mode == 'C':
            real = real * mask_real - imag * mask_imag
            imag = real * mask_imag + imag * mask_real
        elif self.masking_mode == 'R':
            real = real * mask_real
            imag = imag * mask_imag

        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = out_wav.clamp_(-1, 1)
        return out_wav


def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps)
    return torch.mean(snr)


def loss(inputs, label):
    return -(si_snr(inputs, label))


if __name__ == '__main__':
    test_model = DCCRN(rnn_hidden=256, masking_mode='E', use_clstm=True, kernel_num=(32, 64, 128, 256, 256, 256))
    from BaseModel import *

    model_test_timer(test_model, (1, 16000 * 30))
