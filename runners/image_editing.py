import os
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as tvu

from models.diffusion import Model
from functions.process_data import *


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *,
                                               model,
                                               logvar,
                                               betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


def low_pass_filter_2d(tensor, ratio):
    c, h, w = tensor.shape
    assert h == w, f"{tensor.shape}"

    assert 0 <= ratio <= 1., f"ratio must be [0, 1], but {ratio} is given."
    
    fft_tensor = torch.fft.fft2(tensor)
    c_pos = h // 2

    diff = int(c_pos * ratio)
    fft_img_low = fft_tensor
    fft_img_low[:, diff:-diff, diff:-diff] = 0.
    img_low = torch.fft.ifft2(fft_img_low)

    return img_low.real


def high_pass_filter_2d(tensor, ratio):
    c, h, w = tensor.shape
    assert h == w, f"{tensor.shape}"

    assert 0 <= ratio <= 1., f"ratio must be [0, 1], but {ratio} is given."
    
    fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor), dim=(-2, -1))
    c_pos = h // 2

    diff = int(c_pos * ratio)
    fft_img_high = fft_tensor
    fft_img_high[:, c_pos-diff:c_pos+diff, c_pos-diff:c_pos+diff] = 0.
    img_high = torch.fft.ifft2(torch.fft.ifftshift(fft_img_high, dim=(-2, -1)))

    return img_high.real


def random_masking(tensor, ratio):
    mask = np.random.random(size=tensor.shape[-2:])
    
    ret = tensor.clone()
    ret[:, mask > ratio] = 0

    return ret

def create_various_inputs(path, replicate=1):
    import numpy as np
    import torchvision.transforms as T
    from PIL import Image

    with torch.no_grad():
        orig_img = Image.open(path)

        img_list = []

        base_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
        ])
        base_img = base_transform(orig_img)
        base_tensor = T.ToTensor()(base_img)

        # raw image
        img_list.append(base_tensor)

        if replicate > 1:
            for i in range(replicate - 1):
                img_list.append(base_tensor)
            
            return torch.stack(img_list)

        # downsampled
        # resize it to low resolusion and back
        img_list.append(T.Compose([T.Resize(128), T.Resize(256), T.ToTensor()])(base_img)) # 2x
        img_list.append(T.Compose([T.Resize(64), T.Resize(256), T.ToTensor()])(base_img)) # 4x
        img_list.append(T.Compose([T.Resize(32), T.Resize(256), T.ToTensor()])(base_img)) # 8x
        img_list.append(T.Compose([T.Resize(16), T.Resize(256), T.ToTensor()])(base_img)) # 16x

        # low & high pass filter
        img_list.append(low_pass_filter_2d(base_tensor, 0.01))
        img_list.append(low_pass_filter_2d(base_tensor, 0.05))
        img_list.append(low_pass_filter_2d(base_tensor, 0.2))
        # img_list.append(high_pass_filter_2d(base_tensor, 0.02))
        # img_list.append(high_pass_filter_2d(base_tensor, 0.05))
        # img_list.append(high_pass_filter_2d(base_tensor, 0.2))

        # random erase
        # img_list.append(T.Compose([T.RandomErasing(p=1., scale=(0.1, 0.33), ratio=(1, 1))])(base_tensor))
        # img_list.append(T.Compose([T.RandomErasing(p=1., scale=(0.1, 0.33), ratio=(1, 1))])(base_tensor))

        # randomly erase each pixel
        # img_list.append(random_masking(base_tensor, 0.3))
        # img_list.append(random_masking(base_tensor, 0.5))
        # img_list.append(random_masking(base_tensor, 0.7))

        return torch.stack(img_list)

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        self.image_path = args.image_path
        if self.image_path is not None:
            assert os.path.exists(self.image_path), f"{self.image_path} doesn't exist"
        
        self.gif_interval = args.gif_interval

    def image_editing_sample(self):
        print("Loading model")
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        else:
            raise ValueError

        model = Model(self.config)
        ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
        model.load_state_dict(ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        print("Model loaded")
        ckpt_id = 0

        download_process_data(path="colab_demo")
        n = self.config.sampling.batch_size
        model.eval()
        print("Start sampling")
        with torch.no_grad():
            if self.image_path is None:
                name = self.args.npy_name
                [mask, img] = torch.load("colab_demo/{}.pth".format(name))

                mask = mask.to(self.config.device)
                img = img.to(self.config.device)
                img = img.unsqueeze(dim=0)
                img = img.repeat(n, 1, 1, 1)
                x0 = img
            else:
                # img = create_various_inputs(self.image_path)
                img = create_various_inputs(self.image_path, replicate=8)
                mask = torch.zeros(img.shape[1:])
                
                img = img.to(self.config.device)
                mask = mask.to(self.config.device)

                x0 = img

            tvu.save_image(x0, os.path.join(self.args.image_folder, f'original_input.png'))
            x0 = (x0 - 0.5) * 2.

            giffing = [] 

            for it in range(self.args.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.args.t
                a = (1 - self.betas).cumprod(dim=0)

                # create sequence from the input image to noise
                if self.gif_interval > 0:
                    joined_x = tvu.make_grid(x0, nrow=8, padding=10)
                    joined_npy = ((joined_x + 1) * 127.5).detach().cpu().numpy()
                    joined_npy = np.clip(joined_npy, 0, 255).astype(np.uint8)
                    giffing.append(np.transpose(joined_npy, [1, 2, 0]))
                    for t in range(self.gif_interval, total_noise_levels, self.gif_interval):
                        x = x0 * a[t - 1].sqrt() + e * (1.0 - a[t - 1]).sqrt()
                        joined_x = tvu.make_grid(x, nrow=8, padding=10)
                        joined_npy = ((joined_x + 1) * 127.5).detach().cpu().numpy()
                        joined_npy = np.clip(joined_npy, 0, 255).astype(np.uint8)
                        giffing.append(np.transpose(joined_npy, [1, 2, 0]))

                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f'init_{ckpt_id}.png'))

                with tqdm(total=total_noise_levels, desc="Iteration {}".format(it)) as progress_bar:
                    for i in reversed(range(total_noise_levels)):
                        t = (torch.ones(x0.shape[0]) * i).to(self.device)
                        x_ = image_editing_denoising_step_flexible_mask(x, t=t, model=model,
                                                                        logvar=self.logvar,
                                                                        betas=self.betas)
                        x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                        x[:, (mask != 1.)] = x_[:, (mask != 1.)]

                        # reconstructed image at each noise scale
                        x_rec = x / a[i].sqrt() - (1.0 - a[i]).sqrt() * e / a[i].sqrt()


                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'noise_t_{i}_{it}.png'))

                            tvu.save_image((x_rec + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                           f'recon_t_{i}_{it}.png'))

                        if self.gif_interval > 0:
                            interval = self.gif_interval
                            if i < 100:
                                interval /= 2
                            if (i - 99) % interval == 0:
                                joined_x = tvu.make_grid(x, nrow=8, padding=10)
                                joined_npy = ((joined_x + 1) * 127.5).detach().cpu().numpy()
                                joined_npy = np.clip(joined_npy, 0, 255).astype(np.uint8)
                                giffing.append(np.transpose(joined_npy, [1, 2, 0]))
                            
                        progress_bar.update(1)

                x0[:, (mask != 1.)] = x[:, (mask != 1.)]
                torch.save(x, os.path.join(self.args.image_folder,
                                           f'samples_{it}.pth'))
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'samples_{it}.png'))
        
        if self.gif_interval > 0:
            import moviepy.editor as mp
            clip = mp.ImageSequenceClip(giffing, fps=10)
            clip.write_gif("images.gif", fps=10)
            clip.write_videofile("images.mp4")
