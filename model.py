
import torch
from torch import nn

class InstaGAN(object):
    """
    InstaGANの2組のGeneratorとDiscriminatorを保持するためのクラスです。
    """
    def __init__(self, params):
        super(InstaGAN, self).__init__()

        network_params = params["network"]
        common_params = params["common"]

        input_nc = network_params["input_channels"]
        output_nc = network_params["output_channels"]
        norm_layer_name = network_params["norm_layer"]
        assert norm_layer_name in ["batch_norm", "instance_norm"]
        if norm_layer_name == "batch_norm":
            norm_layer = nn.BatchNorm2d
        elif norm_layer_name == "instance_norm":
            norm_layer = nn.InstanceNorm2d
        use_dropout = network_params["use_dropout"]
        n_blocks = network_params["num_resblocks"]
        padding_type = network_params["padding_type"]
        assert padding_type in ["reflect", "replicate", "zeros"]
        use_bias = network_params["use_bias"]
        use_deconv = network_params["use_deconv"]
        self.G_XY = Generator(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
        self.G_YX = Generator(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
        self.D_X = Discriminator(input_nc, norm_layer=norm_layer)
        self.D_Y = Discriminator(input_nc, norm_layer=norm_layer)
    
    def cast_device(self, device):
        self.G_XY.to(device)
        self.G_YX.to(device)
        self.D_X.to(device)
        self.D_Y.to(device)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, 
                       norm_layer=nn.InstanceNorm2d, use_dropout=False,
                       n_blocks=6, padding_type='reflect', use_bias=False, use_deconv=False):
        super(Generator, self).__init__()

        n_downsampling = 2
        self.input_nc = input_nc
        self.image_encoder = Encoder(input_nc, n_downsampling, ngf,
                                    norm_layer, use_dropout, n_blocks, padding_type, use_bias)
        self.mask_encoder = Encoder(1, n_downsampling, ngf,
                                    norm_layer, use_dropout, n_blocks, padding_type, use_bias)
        self.image_decoder = Decoder(output_nc, n_downsampling, ngf * 2, norm_layer, use_bias, use_deconv=use_deconv)
        self.mask_decoder = MaskDecoder(1, n_downsampling, ngf * 3, norm_layer, use_bias, use_deconv=use_deconv)

    def forward(self, image_masks):
        image = image_masks[:, :self.input_nc, :, :]
        masks = image_masks[:, self.input_nc:, :, :]
        mask_existence = (masks).sum(0).sum(-1).sum(-1)
        if mask_existence.sum() <= 0:
            mask_existence[0] = 1
        
        # encoder
        """  
        n_masks = masks.size(1)
        feature_image = self.image_encoder(image)

        feature_masks_list = []
        for i in range(n_masks):
            if mask_existence[i] <= 0:
                continue
            mask = masks[:, i, :, :].unsqueeze(1)
            feature_masks_list.append(self.mask_encoder(mask))
        #feature_masks = torch.cat(feature_masks_list, dim=1)
        feature_masks_sum = torch.zeros_like(feature_image)
        for feature_mask in feature_masks_list:
            feature_masks_sum += feature_mask

        image_out = self.image_decoder(torch.cat([feature_image, feature_masks_sum], dim=1))
        masks_out = torch.zeros_like(masks)

        for i, feature_mask in enumerate(feature_masks_list):
            if mask_existence[i] <= 0:
                masks_out[:, i, :, :] = masks[:, i, :, :].unsqueeze(1)
            else:
                #feature_mask = feature_masks_list[i]
                feature_for_decode_mask = torch.cat([feature_image, feature_mask, feature_masks_sum], dim=1)
                masks_out[:, i, :, :] = (self.mask_decoder(feature_for_decode_mask)).unsqueeze(1)
        #masks_out = torch.cat(masks_out, dim=1)
        out = torch.cat([image_out, masks_out], dim=1)
        """

        feature_image = self.image_encoder(image)
        masks = masks[:, torch.where(mask_existence > 0)[0], :, :]
        b, n_masks, h, w = masks.size()
        masks_ = torch.reshape(masks, (b * n_masks, 1, h, w))

        feature_masks = self.mask_encoder(masks_)     
        b_n, c, h_, w_ = feature_masks.size()
        feature_masks_sum = torch.sum(feature_masks, dim=0, keepdim=True)
        #feature_masks_list = torch.split(feature_masks, 1, dim=0)



        #decoder
        #feature_for_decode_image = torch.cat([feature_image, sum_feature_masks], dim=1)
        image_out = self.image_decoder(torch.cat([feature_image, feature_masks_sum], dim=1))
        #masks_out = []
        feature_image = feature_image.repeat(n_masks, 1, 1, 1)
        feature_masks_sum = feature_masks_sum.repeat(n_masks, 1, 1, 1)
        feature_for_decode_mask = torch.cat([feature_image,
                                             feature_masks,
                                             feature_masks_sum], dim=1)
        masks_out = self.mask_decoder(feature_for_decode_mask)
        masks_out = torch.reshape(masks_out, (b, n_masks, h, w))      
        out = torch.cat([image_out, masks_out], dim=1)

        return out


class Encoder(nn.Module):
    def __init__(self, input_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias):
        super(Encoder, self).__init__()
        self.init_block = nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                         norm_layer(ngf),
                                         nn.ReLU(True))
        self.downsampling_conv_block = nn.ModuleList([])
        mult = 1
        for _ in range(n_downsampling):

            self.downsampling_conv_block += nn.ModuleList([nn.Conv2d(ngf * mult, ngf * mult * 2, 
                                                               kernel_size=3, stride=2, padding=1, bias=use_bias),
                                                     norm_layer(ngf * mult * 2),
                                                     nn.ReLU(True)])
            mult *= 2
        self.downsampling_conv_block = nn.Sequential(*self.downsampling_conv_block)
        self.resnet_block = nn.ModuleList([])
        for i in range(n_blocks):
            self.resnet_block.append(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                              use_dropout=use_dropout, use_bias=use_bias))
        self.resnet_block = nn.Sequential(*self.resnet_block)
    
    def forward(self, image):
        image = self.init_block(image)
        image = self.downsampling_conv_block(image)
        image = self.resnet_block(image)
        return image
        

class Decoder(nn.Module):
    def __init__(self, output_nc, n_downsampling, ngf, norm_layer, use_bias, use_deconv=True):
        super(Decoder, self).__init__()        
        mult = 2**n_downsampling
        self.decoder = nn.ModuleList([nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                                   norm_layer(ngf * mult),
                                                   nn.ReLU(True)])

        for _ in range(n_downsampling):
            if use_deconv:
                self.decoder += nn.ModuleList([nn.ConvTranspose2d(ngf * mult, (ngf * mult) // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                                                norm_layer((ngf * mult) // 2),
                                                nn.ReLU(True)])
            else:
                self.decoder += nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'),
                                                   nn.Conv2d(ngf * mult, (ngf * mult) // 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                                   norm_layer((ngf * mult) // 2),
                                                   nn.ReLU(True)])
            mult //= 2
        
        self.final_block = nn.ModuleList([nn.ReflectionPad2d(3),
                                          nn.Conv2d(ngf * mult, output_nc, kernel_size=7),
                                          nn.Tanh()])

        self.decoder = nn.Sequential(*self.decoder)
        self.final_block = nn.Sequential(*self.final_block)
    
    def forward(self, features):
        x = self.decoder(features)
        image = self.final_block(x)
        return image


class MaskDecoder(nn.Module):
    def __init__(self, output_nc, n_downsampling, ngf, norm_layer, use_bias, use_deconv=True):
        super(MaskDecoder, self).__init__()        
        mult = 2**n_downsampling
        self.decoder = nn.ModuleList([nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                                   norm_layer(ngf * mult),
                                                   nn.ReLU(True)])

        for _ in range(n_downsampling):
            if use_deconv:
                self.decoder += nn.ModuleList([nn.ConvTranspose2d(ngf * mult, (ngf * mult) // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                                                norm_layer((ngf * mult) // 2),
                                                nn.ReLU(True)])
            else:
                self.decoder += nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'),
                                                   nn.Conv2d(ngf * mult, (ngf * mult) // 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                                   norm_layer((ngf * mult) // 2),
                                                   nn.ReLU(True)])
            mult //= 2
        
        self.final_block = nn.ModuleList([nn.ReflectionPad2d(3),
                                          nn.Conv2d(ngf * mult, output_nc, kernel_size=7),
                                          nn.Sigmoid()])

        self.decoder = nn.Sequential(*self.decoder)
        self.final_block = nn.Sequential(*self.final_block)
    
    def forward(self, features):
        x = self.decoder(features)
        image = self.final_block(x)
        return image

class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=32, n_layers=3, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(Discriminator, self).__init__()
        self.input_nc = input_nc

        kw = 4
        padw = 1
        self.image_encoder = self._get_feature_extractor(input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias)
        self.mask_encoder = self._get_feature_extractor(1, ndf, n_layers, kw, padw, norm_layer, use_bias)
        self.classifier = self._get_classifier(2 * ndf, n_layers, kw, padw, norm_layer)  # 2*ndf

    def _get_feature_extractor(self, input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias):
        model = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                # Use spectral normalization
                SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        return nn.Sequential(*model)

    def _get_classifier(self, ndf, n_layers, kw, padw, norm_layer):
        nf_mult_prev = min(2 ** (n_layers-1), 8)
        nf_mult = min(2 ** n_layers, 8)
        model = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # Use spectral normalization
        model += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        return nn.Sequential(*model)


    def forward(self, image_masks):
        # split data
        image = image_masks[:, :self.input_nc, :, :]  # (B, CX, W, H)
        masks = image_masks[:, self.input_nc:, :, :]  # (B, CA, W, H)
        mask_existence = (masks).mean(0).mean(-1).mean(-1)
        if mask_existence.sum() <= 0:
            mask_existence[0] = 1  # forward at least one segmentation

        # run feature extractor
        feature_image = self.image_encoder(image)

        """
        feature_masks = list()
        for i in range(masks.size(1)):
            if mask_existence[i] > 0:  # skip empty segmentation
                mask = masks[:, i, :, :].unsqueeze(1)
                feature_masks.append(self.mask_encoder(mask))
        feature_masks_sum = torch.sum(torch.stack(feature_masks), dim=0)  # aggregated set feature
        """

        masks = masks[:, torch.where(mask_existence > 0)[0], :, :]
        b, n_masks, h, w = masks.size()
        masks = torch.reshape(masks, (b * n_masks, 1, h, w))
        feature_masks = self.mask_encoder(masks)     
        feature_masks_sum = torch.sum(feature_masks, dim=0, keepdim=True)

        # run classifier
        feature = torch.cat([feature_image, feature_masks_sum], dim=1)
        out = self.classifier(feature)
        return out

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_1 = nn.Sequential(self._get_padding(padding_type, 1),
                                     nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
                                     norm_layer(dim),
                                     nn.ReLU(True))
        if use_dropout:
            self.conv_1.append(nn.Dropout(0.5))

        self.conv_2 = nn.Sequential(self._get_padding(padding_type, 1),
                                     nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
                                     norm_layer(dim),
                                     )
        self.relu = nn.ReLU(True)
    def forward(self, x):
        y = self.conv_2(self.conv_1(x))
        return self.relu(x + y)
    
    def _get_padding(self, padding_type, pad=1):
        if padding_type == "reflect":
            return nn.ReflectionPad2d(pad)
        elif padding_type == "replicate":
            return nn.ReplicationPad2d(pad)
        elif padding_type == "zeros":
            return nn.ZeroPad2d(pad)
        else:
            raise NotImplementedError(f"padding {padding_type} is not implemented")


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)