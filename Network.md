class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1):
        super().__init__()

        if padding is None:
            padding = ((kernel_size - 1) // 2) * dilation

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        max_out = torch.amax(x, dim=(2, 3), keepdim=True)

        out = self.fc1(avg_out) + self.fc1(max_out)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)

        return torch.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)

        return torch.sigmoid(out) * x


class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)

        return x * self.sigmoid(y)


class GatedECAFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.refine = DepthwiseSeparableConv(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            dilation=1
        )

        self.eca = ECAAttention(kernel_size=3)

    def forward(self, pvt_feat, eff_feat):
        gate = self.gate(torch.cat([pvt_feat, eff_feat], dim=1))
        fused = gate * pvt_feat + (1.0 - gate) * eff_feat
        fused = self.refine(fused)
        fused = self.eca(fused)

        return fused


class CrossSemanticAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        inter_channels = max(channels // 2, 1)

        self.theta = nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False)
        self.g = nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False)
        self.out_conv = nn.Conv2d(inter_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, _, h, w = x.size()

        theta_x = self.theta(x).view(b, -1, h * w)
        phi_x = self.phi(x).view(b, -1, h * w)
        g_x = self.g(x).view(b, -1, h * w)

        attention = torch.matmul(theta_x.transpose(1, 2), phi_x)
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, g_x.transpose(1, 2)).transpose(1, 2)
        out = out.view(b, -1, h, w)
        out = self.out_conv(out)

        return out + x


class FeatureEnhancementModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_nonlocal=True):
        super().__init__()

        self.conv_dilated1 = ConvBNAct(in_channels, out_channels, kernel_size=3, dilation=1)
        self.conv_dilated3 = ConvBNAct(in_channels, out_channels, kernel_size=3, dilation=3)
        self.conv_dilated5 = ConvBNAct(in_channels, out_channels, kernel_size=3, dilation=5)

        self.csa1 = CrossSemanticAttention(out_channels)
        self.csa2 = CrossSemanticAttention(out_channels)
        self.csa3 = CrossSemanticAttention(out_channels)

        self.final_conv = ConvBNAct(out_channels * 3, out_channels, kernel_size=3)

        if use_nonlocal:
            self.non_local = NonLocalBlock(out_channels)
        else:
            self.non_local = nn.Identity()

        if in_channels != out_channels:
            self.residual_proj = ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        out1 = self.csa1(self.conv_dilated1(x))
        out2 = self.csa2(self.conv_dilated3(x))
        out3 = self.csa3(self.conv_dilated5(x))

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.final_conv(out)

        residual = self.residual_proj(x)
        out = out + residual
        out = self.non_local(out)

        return out


class ShapeAwareModule(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.edge_conv = ConvBNAct(channels, channels, kernel_size=3)
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
        self.fuse = ConvBNAct(channels * 3, channels, kernel_size=3)

    def forward(self, e1, e2, e3):
        edge1 = self.edge_conv(e1)
        edge2 = self.edge_conv(e2)
        edge3 = self.edge_conv(e3)

        d12 = torch.abs(edge1 - edge2)
        d23 = torch.abs(edge2 - edge3)
        d13 = torch.abs(edge1 - edge3)

        d12 = self.channel_attention(d12)
        d23 = self.channel_attention(d23)
        d13 = self.channel_attention(d13)

        fused = torch.cat([d12, d23, d13], dim=1)
        fused = self.fuse(fused)
        fused = self.spatial_attention(fused)

        return fused


class MultiBranchAttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = DepthwiseSeparableConv(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            dilation=1
        )

        self.branch2 = DepthwiseSeparableConv(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=2,
            dilation=2
        )

        self.branch3 = DepthwiseSeparableConv(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=3,
            dilation=3
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        residual = x

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        out = torch.cat([b1, b2, b3], dim=1)
        out = self.fuse(out)

        out = self.channel_att(out)
        out = self.spatial_att(out)

        return out + residual


class FinalSegmentationModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.PVTV2 = timm.create_model(
            'pvt_v2_b1',
            pretrained=True,
            features_only=True
        )

        for param in self.PVTV2.parameters():
            param.requires_grad = False

        self.efficientnet = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            features_only=True
        )

        self.conv1x1_pvt56 = ConvBNAct(64, 96, kernel_size=1, padding=0)
        self.conv1x1_pvt28 = ConvBNAct(128, 128, kernel_size=1, padding=0)
        self.conv1x1_pvt14 = ConvBNAct(320, 192, kernel_size=1, padding=0)
        self.conv1x1_pvt7 = ConvBNAct(512, 256, kernel_size=1, padding=0)

        self.conv1x1_eff56 = ConvBNAct(24, 96, kernel_size=1, padding=0)
        self.conv1x1_eff28 = ConvBNAct(40, 128, kernel_size=1, padding=0)
        self.conv1x1_eff14 = ConvBNAct(112, 192, kernel_size=1, padding=0)
        self.conv1x1_eff7 = ConvBNAct(320, 256, kernel_size=1, padding=0)

        self.fusion56 = GatedECAFusion(channels=96)
        self.fusion28 = GatedECAFusion(channels=128)
        self.fusion14 = GatedECAFusion(channels=192)
        self.fusion7 = GatedECAFusion(channels=256)

        self.reduce_fusion14 = ConvBNAct(192 + 256, 192, kernel_size=1, padding=0)
        self.reduce_fusion28 = ConvBNAct(128 + 192, 128, kernel_size=1, padding=0)
        self.reduce_fusion56 = ConvBNAct(96 + 128, 96, kernel_size=1, padding=0)

        self.fem_28 = FeatureEnhancementModule(
            in_channels=128,
            out_channels=128,
            use_nonlocal=False
        )

        self.fem_14 = FeatureEnhancementModule(
            in_channels=192,
            out_channels=192,
            use_nonlocal=False
        )

        self.fem_7 = FeatureEnhancementModule(
            in_channels=256,
            out_channels=256,
            use_nonlocal=True
        )

        self.sam_proj28 = ConvBNAct(128, 96, kernel_size=1, padding=0)
        self.sam_proj14 = ConvBNAct(192, 96, kernel_size=1, padding=0)
        self.shape_aware = ShapeAwareModule(96)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder4_fuse = MultiBranchAttentionFusion(in_channels=448)
        self.conv_decoder4 = ConvBNAct(448, 192, kernel_size=1, padding=0)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder3_fuse = MultiBranchAttentionFusion(in_channels=320)
        self.conv_decoder3 = ConvBNAct(320, 256, kernel_size=1, padding=0)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder2_fuse = MultiBranchAttentionFusion(in_channels=352)
        self.conv_decoder2 = ConvBNAct(352, 192, kernel_size=1, padding=0)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder1_fuse = MultiBranchAttentionFusion(in_channels=192)
        self.conv_decoder1 = ConvBNAct(192, 96, kernel_size=1, padding=0)

        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_decoder0 = ConvBNAct(96, 64, kernel_size=1, padding=0)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        pvt_features = self.PVTV2(x)
        eff_features = self.efficientnet(x)

        pvt_features56 = self.conv1x1_pvt56(pvt_features[-4])
        pvt_features28 = self.conv1x1_pvt28(pvt_features[-3])
        pvt_features14 = self.conv1x1_pvt14(pvt_features[-2])
        pvt_features7 = self.conv1x1_pvt7(pvt_features[-1])

        eff_features56 = self.conv1x1_eff56(eff_features[-4])
        eff_features28 = self.conv1x1_eff28(eff_features[-3])
        eff_features14 = self.conv1x1_eff14(eff_features[-2])
        eff_features7 = self.conv1x1_eff7(eff_features[-1])

        combined56 = self.fusion56(pvt_features56, eff_features56)
        combined28 = self.fusion28(pvt_features28, eff_features28)
        combined14 = self.fusion14(pvt_features14, eff_features14)
        combined7 = self.fusion7(pvt_features7, eff_features7)

        combined7 = self.fem_7(combined7)

        upsampled7 = F.interpolate(
            combined7,
            size=combined14.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        combined14 = self.reduce_fusion14(torch.cat([combined14, upsampled7], dim=1))
        combined14 = self.fem_14(combined14)

        upsampled14 = F.interpolate(
            combined14,
            size=combined28.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        combined28 = self.reduce_fusion28(torch.cat([combined28, upsampled14], dim=1))
        combined28 = self.fem_28(combined28)

        upsampled28 = F.interpolate(
            combined28,
            size=combined56.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        combined56 = self.reduce_fusion56(torch.cat([combined56, upsampled28], dim=1))

        sam_e1 = combined56
        sam_e2 = F.interpolate(
            self.sam_proj28(combined28),
            size=combined56.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        sam_e3 = F.interpolate(
            self.sam_proj14(combined14),
            size=combined56.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        shape_guidance = self.shape_aware(sam_e1, sam_e2, sam_e3)
        combined56 = combined56 + shape_guidance

        d4 = self.upsample4(combined7)
        combined14_skip = F.interpolate(
            combined14,
            size=d4.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        d4 = torch.cat([d4, combined14_skip], dim=1)
        d4 = self.decoder4_fuse(d4)
        d4 = self.conv_decoder4(d4)

        d3 = self.upsample3(d4)
        combined28_skip = F.interpolate(
            combined28,
            size=d3.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        d3 = torch.cat([d3, combined28_skip], dim=1)
        d3 = self.decoder3_fuse(d3)
        d3 = self.conv_decoder3(d3)

        d2 = self.upsample2(d3)
        combined56_skip = F.interpolate(
            combined56,
            size=d2.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        d2 = torch.cat([d2, combined56_skip], dim=1)
        d2 = self.decoder2_fuse(d2)
        d2 = self.conv_decoder2(d2)

        d1 = self.upsample1(d2)
        d1 = self.decoder1_fuse(d1)
        d1 = self.conv_decoder1(d1)

        d0 = self.upsample0(d1)
        d0 = self.conv_decoder0(d0)

        return self.final_conv(d0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = FinalSegmentationModel(num_classes=1).to(device)

print("Model architecture:")
print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)
