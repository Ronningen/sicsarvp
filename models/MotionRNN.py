__author__ = 'haixu'
__author__ = 'yunbo'

import torch
import torch.nn as nn


class Warp(nn.Module):
    def __init__(self, inc, outc, neighbour=3):
        super(Warp, self).__init__()
        self.neighbour = neighbour
        self.zero_padding = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(inc, outc, kernel_size=neighbour, stride=neighbour, bias=None)
        self.warp_gate = nn.Conv2d(inc, neighbour * neighbour, kernel_size=3, padding=1, stride=1)
        nn.init.constant_(self.warp_gate.weight, 0)
        self.warp_gate.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, info):
        x = info[0]
        offset = info[1]

        dtype = offset.data.type()
        N = self.neighbour * self.neighbour

        m = torch.sigmoid(self.warp_gate(x))
        x = self.zero_padding(x)
        ## Neighbourhood Warp Operation
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_warped = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        ## Warp Gate
        m = m.contiguous().permute(0, 2, 3, 1)
        m = m.unsqueeze(dim=1)
        m = torch.cat([m for _ in range(x_warped.size(1))], dim=1)
        x_warped *= m

        x_warped = self._reshape_x_warped(x_warped, self.neighbour)
        out = self.conv(x_warped)
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.neighbour - 1) // 2, (self.neighbour - 1) // 2 + 1),
            torch.arange(-(self.neighbour - 1) // 2, (self.neighbour - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(1, h + 1, 1), torch.arange(1, w + 1, 1))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_warped = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_warped

    @staticmethod
    def _reshape_x_warped(x_warped, neighbour):
        b, c, h, w, N = x_warped.size()
        x_warped = torch.cat(
            [x_warped[..., s:s + neighbour].contiguous().view(b, c, h, w * neighbour) for s in range(0, N, neighbour)],
            dim=-1)
        x_warped = x_warped.contiguous().view(b, c, h * neighbour, w * neighbour)
        return x_warped


class MotionGRU(nn.Module):
    def __init__(self, in_channel, motion_hidden, neighbour):
        super(MotionGRU, self).__init__()
        self.update = nn.Conv2d(in_channel + motion_hidden, motion_hidden, kernel_size=3, stride=1, padding=1)
        nn.init.constant_(self.update.weight, 0)
        self.update.register_full_backward_hook(self._set_lr)

        self.reset = nn.Conv2d(in_channel + motion_hidden, motion_hidden, kernel_size=3, stride=1, padding=1)
        nn.init.constant_(self.reset.weight, 0)
        self.reset.register_full_backward_hook(self._set_lr)

        self.output = nn.Conv2d(in_channel + motion_hidden, motion_hidden, kernel_size=3, stride=1, padding=1)
        nn.init.constant_(self.output.weight, 0)
        self.output.register_full_backward_hook(self._set_lr)

        self.warp = Warp(in_channel, in_channel, neighbour)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x_t, pre_offset, mean):
        stacked_inputs = torch.cat([x_t, pre_offset], dim=1)
        update_gate = torch.sigmoid(self.update(stacked_inputs))
        reset_gate = torch.sigmoid(self.reset(stacked_inputs))
        offset = torch.tanh(self.output(torch.cat([x_t, pre_offset * reset_gate], dim=1)))
        offset = pre_offset * (1 - update_gate) + offset * update_gate
        mean = mean + 0.5 * (pre_offset - mean)
        offset = offset + mean

        x_t = self.warp([x_t, offset])
        return x_t, offset, mean


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self._forget_bias = 1.0
        padding = filter_size // 2

        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, filter_size, stride, padding, bias=False),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, filter_size, stride, padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, filter_size, stride, padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, filter_size, stride, padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, filter_size, stride, padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, filter_size, stride, padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, filter_size, stride, padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, filter_size, stride, padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t, motion_highway):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        m_new_new = self.conv_last(mem)

        # Motion Highway
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(m_new_new) + (1 - o_t) * motion_highway
        motion_highway = h_new
        return h_new, c_new, m_new, motion_highway


class RNN(nn.Module):
    def __init__(self, img_height, img_width, img_channel, num_layers=4, num_hidden=[64,64,64,64], filter_size=5, device=None):
        super(RNN, self).__init__()
        self.patch_height = img_height
        self.patch_width = img_width
        self.patch_ch = img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.neighbour = 3
        self.motion_hidden = 2 * self.neighbour * self.neighbour

        # self.w_w = w_w
        # self.i_w = i_w
        # self.e_w = e_w
        # self.starter = starter

        self.device = device

        cell_list = []
        for i in range(num_layers):
            in_channel = self.patch_ch if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], self.patch_height, self.patch_width, filter_size, 1, 1),
            )
        enc_list = []
        for i in range(num_layers - 1):
            enc_list.append(
                nn.Conv2d(num_hidden[i], num_hidden[i] // 4, kernel_size=filter_size, stride=2, padding=filter_size // 2),
            )
        motion_list = []
        for i in range(num_layers - 1):
            motion_list.append(
                MotionGRU(num_hidden[i] // 4, self.motion_hidden, self.neighbour)
            )
        dec_list = []
        for i in range(num_layers - 1):
            dec_list.append(
                nn.ConvTranspose2d(num_hidden[i] // 4, num_hidden[i], kernel_size=4, stride=2, padding=1),
            )
        gate_list = []
        for i in range(num_layers - 1):
            gate_list.append(
                nn.Conv2d(num_hidden[i] * 2, num_hidden[i], kernel_size=filter_size, stride=1,  padding=filter_size // 2),
            )
        self.gate_list = nn.ModuleList(gate_list)
        self.cell_list = nn.ModuleList(cell_list)
        self.motion_list = nn.ModuleList(motion_list)
        self.enc_list = nn.ModuleList(enc_list)
        self.dec_list = nn.ModuleList(dec_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.patch_ch, 1, stride=1, padding=0, bias=False)
        self.conv_first_v = nn.Conv2d(self.patch_ch, num_hidden[0], 1, stride=1, padding=0, bias=False)

        self.to(device)

    def fullforward(self, frames):
        # [batch, time, channel, height, width]
        b, t, c, h, w = frames.shape

        next_frames = []
        h_t = []
        c_t = []
        h_t_conv = []
        h_t_conv_offset = []
        mean = []

        for i in range(self.num_layers):
            zeros = torch.empty([b, self.num_hidden[i], h, w], device=self.device)
            nn.init.xavier_normal_(zeros)
            h_t.append(zeros)
            c_t.append(zeros)

        for i in range(self.num_layers - 1):
            zeros = torch.empty([b, self.num_hidden[i] // 4, h // 2, w // 2], device=self.device)
            nn.init.xavier_normal_(zeros)
            h_t_conv.append(zeros)
            zeros = torch.empty([b, self.motion_hidden, h // 2, w // 2], device=self.device)
            nn.init.xavier_normal_(zeros)
            h_t_conv_offset.append(zeros)
            mean.append(zeros)

        mem = torch.empty([b, self.num_hidden[0], h, w], device=self.device)
        motion_highway = torch.empty([b, self.num_hidden[0], h, w], device=self.device)
        nn.init.xavier_normal_(mem)
        nn.init.xavier_normal_(motion_highway)

        for t in range(self.w_w + self.i_w + self.e_w):
            if t == 0:
                net = frames[:, t]
            elif t < self.w_w + self.i_w + 1:
                net = frames[:, t]
                net0 = net[:,:1] + x_gen[:,:1]*(net[:,:1]<=0)
                net = torch.cat((net0, net[:,1:]),1)
            else:
                net = x_gen

            x_gen, h_t, c_t, h_t_conv, h_t_conv_offset, mean, mem = self.step(net, h_t, c_t, h_t_conv, h_t_conv_offset, mean, mem)

            motion_highway = self.conv_first_v(net)
            h_t[0], c_t[0], mem, motion_highway = self.cell_list[0](net, h_t[0], c_t[0], mem, motion_highway)
            net = self.enc_list[0](h_t[0])
            h_t_conv[0], h_t_conv_offset[0], mean[0] = self.motion_list[0](net, h_t_conv_offset[0], mean[0])
            h_t_tmp = self.dec_list[0](h_t_conv[0])
            o_t = torch.sigmoid(self.gate_list[0](torch.cat([h_t_tmp, h_t[0]], dim=1)))
            h_t[0] = o_t * h_t_tmp + (1 - o_t) * h_t[0]

            for i in range(1, self.num_layers - 1):
                h_t[i], c_t[i], mem, motion_highway = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], mem, motion_highway)
                net = self.enc_list[i](h_t[i])
                h_t_conv[i], h_t_conv_offset[i], mean[i] = self.motion_list[i](net, h_t_conv_offset[i], mean[i])
                h_t_tmp = self.dec_list[i](h_t_conv[i])
                o_t = torch.sigmoid(self.gate_list[i](torch.cat([h_t_tmp, h_t[i]], dim=1)))
                h_t[i] = o_t * h_t_tmp + (1 - o_t) * h_t[i]

            h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem, motion_highway = self.cell_list[
                self.num_layers - 1](
                h_t[self.num_layers - 2], h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem, motion_highway)
            x_gen = self.conv_last(h_t[self.num_layers - 1])

            if t >= self.w_w:
                next_frames.append(x_gen[:,0])

        return torch.stack(next_frames, dim=0).permute(1, 0, 2, 3).contiguous()

    def init(self, frames):
        b, c, h, w = frames.shape
        
        h_t = []
        c_t = []
        h_t_conv = []
        h_t_conv_offset = []
        mean = []

        for i in range(self.num_layers):
            zeros = torch.empty([b, self.num_hidden[i], h, w], device=self.device)
            nn.init.xavier_normal_(zeros)
            # zeros = zeros/100
            h_t.append(zeros)
            c_t.append(zeros)

        for i in range(self.num_layers - 1):
            zeros = torch.empty([b, self.num_hidden[i] // 4, h // 2, w // 2], device=self.device)
            nn.init.xavier_normal_(zeros)
            # zeros = zeros/100
            h_t_conv.append(zeros)
            zeros = torch.empty([b, self.motion_hidden, h // 2, w // 2], device=self.device)
            nn.init.xavier_normal_(zeros)
            # zeros = zeros/100
            h_t_conv_offset.append(zeros)
            mean.append(zeros)

        mem = torch.empty([b, self.num_hidden[0], h, w], device=self.device)
        nn.init.xavier_normal_(mem)
        # mem = mem/100
        
        # motion_highway = torch.empty([b, self.num_hidden[0], h, w], device=self.device)
        # nn.init.xavier_normal_(motion_highway)

        return frames, h_t, c_t, h_t_conv, h_t_conv_offset, mean, mem

    def forward(self, net, h_t, c_t, h_t_conv, h_t_conv_offset, mean, mem):
        motion_highway = self.conv_first_v(net)
        h_t[0], c_t[0], mem, motion_highway = self.cell_list[0](net, h_t[0], c_t[0], mem, motion_highway)
        net = self.enc_list[0](h_t[0])
        h_t_conv[0], h_t_conv_offset[0], mean[0] = self.motion_list[0](net, h_t_conv_offset[0], mean[0])
        h_t_tmp = self.dec_list[0](h_t_conv[0])
        o_t = torch.sigmoid(self.gate_list[0](torch.cat([h_t_tmp, h_t[0]], dim=1)))
        h_t[0] = o_t * h_t_tmp + (1 - o_t) * h_t[0]

        for i in range(1, self.num_layers - 1):
            h_t[i], c_t[i], mem, motion_highway = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], mem, motion_highway)
            net = self.enc_list[i](h_t[i])
            h_t_conv[i], h_t_conv_offset[i], mean[i] = self.motion_list[i](net, h_t_conv_offset[i], mean[i])
            h_t_tmp = self.dec_list[i](h_t_conv[i])
            o_t = torch.sigmoid(self.gate_list[i](torch.cat([h_t_tmp, h_t[i]], dim=1)))
            h_t[i] = o_t * h_t_tmp + (1 - o_t) * h_t[i]

        h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem, motion_highway = self.cell_list[
            self.num_layers - 1](
            h_t[self.num_layers - 2], h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem, motion_highway)
        x_gen = self.conv_last(h_t[self.num_layers - 1])
        return x_gen, h_t, c_t, h_t_conv, h_t_conv_offset, mean, mem
