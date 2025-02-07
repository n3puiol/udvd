import torch
import torch.nn as nn
import torch.nn.functional as F

# import vae
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

# local import
from models import register_model

class crop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-1, 0:W]
        return x

class shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,1,0))
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift_down = nn.ZeroPad2d((0,0,1,0))
            self.crop = crop()
        self.replicate = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, bias=bias)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.blind:
            x = self.shift_down(x)
        x = self.replicate(x)
        x = self.conv(x)
        x = self.relu(x)
        if self.blind:
            x = self.crop(x)
        return x

class Pool(nn.Module):
    def __init__(self, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift = shift()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.blind:
            x = self.shift(x)
        x = self.pool(x)
        return x

class rotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x90 = x.transpose(2,3).flip(3)
        x180 = x.flip(2).flip(3)
        x270 = x.transpose(2,3).flip(2)
        x = torch.cat((x,x90,x180,x270), dim=0)
        return x

class unrotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0, x90, x180, x270 = torch.chunk(x, 4, dim=0)
        x90 = x90.transpose(2,3).flip(2)
        x180 = x180.flip(2).flip(3)
        x270 = x270.transpose(2,3).flip(3)
        x = torch.cat((x0,x90,x180,x270), dim=1)
        return x

class ENC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=False, reduce=True, blind=True):
        super().__init__()
        self.reduce = reduce
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, out_channels, bias=bias, blind=blind)
        if reduce:
            self.pool = Pool(blind=blind)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.reduce:
            x = self.pool(x)
        return x

class DEC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=False, blind=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv4 = Conv(mid_channels, out_channels, bias=bias, blind=blind)

    def forward(self, x, x_in):
        x = self.upsample(x)

        # Smart Padding
        diffY = x_in.size()[2] - x.size()[2]
        diffX = x_in.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat((x, x_in), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Blind_UNet(nn.Module):
    def __init__(self, n_channels=3, n_output=96, bias=False, blind=True):
        super().__init__()
        self.n_channels = n_channels
        self.bias = bias
        self.enc1 = ENC_Conv(n_channels, 48, 48, bias=bias, blind=blind)
        self.enc2 = ENC_Conv(48, 48, 48, bias=bias, blind=blind)
        self.enc3 = ENC_Conv(48, 96, 48, bias=bias, reduce=False, blind=blind)
        self.dec2 = DEC_Conv(96, 96, 96, bias=bias, blind=blind)
        self.dec1 = DEC_Conv(96+n_channels, 96, n_output, bias=bias, blind=blind)

    def forward(self, input):
        x1 = self.enc1(input)
        x2 = self.enc2(x1)
        x = self.enc3(x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, input)
        return x

@register_model("blind-spot-net-4")
class BlindSpotNet(nn.Module):
    def __init__(self, n_channels=3, n_output=9, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.n_channels = n_channels
        self.c = n_channels
        self.n_output = n_output
        self.bias = bias
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.unet = Blind_UNet(n_channels=n_channels, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = Blind_UNet(n_channels=n_channels, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, n_output, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--in-channels", type=int, default=3, help="number of input channels")
        parser.add_argument("--out-channels", type=int, default=9, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(n_channels=args.in_channels, n_output=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        x = self.rotate(x)
        x = self.unet(x)
        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x, sigma

@register_model("blind-video-net-d1-4")
class BlindVideoNetD1(nn.Module):
    def __init__(self, channels_per_frame=3, out_channels=9, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.denoiser_1 = Blind_UNet(n_channels=3*channels_per_frame, n_output=96, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = Blind_UNet(n_channels=3*channels_per_frame, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=3, help="number of channels per frame")
        parser.add_argument("--out-channels", type=int, default=9, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        x = self.rotate(x)
        x = self.denoiser_1(x)
        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x, sigma

# THE UDVD MODEL to adapt
@register_model("blind-video-net-4")
class BlindVideoNet(nn.Module):
    def __init__(self, channels_per_frame=3, out_channels=9, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.denoiser_1 = Blind_UNet(n_channels=3*channels_per_frame, n_output=32, bias=bias, blind=blind)
        self.denoiser_2 = Blind_UNet(n_channels=96, n_output=96, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = Blind_UNet(n_channels=5*channels_per_frame, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.vae.to("cuda")
        self.vae.eval()

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=3, help="number of channels per frame")
        parser.add_argument("--out-channels", type=int, default=9, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def prepare_latents(self, x):
        x = x.to("cuda")
        print("frame splitting")
        # x is assumed to be of shape [B, 9, H, W] where 9 = 3 frames * 3 channels per frame.
        # Split x into three frames along the channel dimension.
        print("f1")
        f1 = x[:, 0:3, :, :]
        print("f1 shape: ", f1.shape)
        print("f2")
        f2 = x[:, 3:6, :, :]
        print("f3")
        f3 = x[:, 6:9, :, :]

        print("frame encoding")
        with torch.no_grad():
            # Encode each frame separately.
            print("f1")
            f1 = f1.to("cuda")
            f1_enc = self.vae.encode(f1).latent_dist.sample()
            torch.cuda.synchronize()  # force sync to catch errors
            print("f1_enc shape: ", f1_enc.shape)
            print("f2")
            f2 = f2.to("cuda")
            f2_enc = self.vae.encode(f2).latent_dist.sample()
            torch.cuda.synchronize()  # force sync to catch errors
            print("f3")
            f3 = f3.to("cuda")
            f3_enc = self.vae.encode(f3).latent_dist.sample()
            torch.cuda.synchronize()  # force sync to catch errors
        
        print("frame concatinating")
        # Concatenate the encoded frames along the channel dimension.
        latents = torch.cat([f1_enc, f2_enc, f3_enc], dim=1)
        return latents

    
    def decode_latents(self, latents):
        print("latent decoding")
        # Assume latents has shape [B, 3 * latent_channels, H_lat, W_lat]
        channels_per_frame = latents.shape[1] // 3
        
        # Split the concatenated latent representation into 3 latent frames.
        latent_frames = [latents[:, i * channels_per_frame:(i + 1) * channels_per_frame, :, :]
                        for i in range(3)]
        
        # Decode each latent frame separately.
        decoded_frames = [self.vae.decode(latent_frame).sample for latent_frame in latent_frames]
        
        # Concatenate the decoded frames along the channel dimension to produce an output with 9 channels.
        denoised_x = torch.cat(decoded_frames, dim=1)
        print("end decoding")
        return denoised_x


    def forward(self, x):

        debug = True

        if debug: print("add padding")
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        i1 = self.rotate(x[:, 0:(3*self.c), :, :])
        i2 = self.rotate(x[:, self.c:(4*self.c), :, :])
        i3 = self.rotate(x[:, (2*self.c):(5*self.c), :, :])
        
        if debug: print("to latent space")
        # image to latent space
        if debug: print("i1 shape:", i1.shape)
        lat_i1 = self.prepare_latents(i1)
        if debug: print("i2 shape:", i2.shape)
        lat_i2 = self.prepare_latents(i2)
        if debug: print("i3 shape:", i3.shape)
        lat_i3 = self.prepare_latents(i3)
        print("lat_i1 shape: ", lat_i1.shape)
        print("lat_i2 shape: ", lat_i2.shape)
        print("lat_i3 shape: ", lat_i3.shape)
        if debug: print("denoiser 1 * 3")
        y1 = self.denoiser_1(lat_i1)
        y2 = self.denoiser_1(lat_i2)
        y3 = self.denoiser_1(lat_i3)

        if debug: print("concat+denoiser 2")
        y = torch.cat((y1, y2, y3), dim=1)
        x = self.denoiser_2(y)

        if debug: print("from latent space")
        # Latent to image space
        reconstructed_x = self.decode_latents(x)

        if debug: print("post process, unrotate")
        if self.blind:
            reconstructed_x = self.shift(reconstructed_x)
        x = self.unrotate(reconstructed_x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)

        if debug: print("remove padding")
        # Unsquare
        if H > W:
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif W > H:
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]

        return x, sigma, reconstructed_x, torch.cat((i1, i2, i3), dim=1)
