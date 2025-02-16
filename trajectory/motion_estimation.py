import os
import os.path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms as transforms
import sys
import data
import utils

sys.path.append('../')
sns.set_theme()


class MotionEstimation:
    def __init__(self):
        self.parallel = True
        self.fast = False
        self.pretrained = True
        self.old = True
        self.load_opt = False

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.transform = transforms.Compose([transforms.ToPILImage()])
        self.to_gray = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1)])

        self.patch_size = 256
        self.stride = 64
        self.is_image = False
        self.n_frames = 5
        self.cpf = 3
        self.mid = self.n_frames // 2
        self.is_real = False

        self.aug = 0

        self.dist = 'G'
        self.mode = 'S'
        self.noise_std = 30
        self.min_noise = 0
        self.max_noise = 100

        self.batch_size = 1
        self.lr = 1e-4

        self.path = "../pretrained/blind_video_net.pt"

        self.model, self.optimizer, self.args = utils.load_model(self.path, parallel=self.parallel,
                                                                 pretrained=self.pretrained, old=self.old,
                                                                 load_opt=self.load_opt)
        self.model.to(self.device)
        print(self.model)

        self.test_loader = None
        self.num = None
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.ps = None
        self.qs = None

    def load_sample(self, video="bus", dataset="DAVIS", num=25, x=481, y=219, w=128, h=128):
        if self.test_loader is None:
            path = os.path.join("../datasets", dataset)
            self.num = num
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            px = 20 * np.arange(0, 3) + 100
            py = 20 * np.arange(0, 3) + 100
            self.ps, self.qs = np.meshgrid(px, py)
            self.ps = np.append(self.ps, 64)
            self.qs = np.append(self.qs, 64)
            train_loader, self.test_loader = data.build_dataset("SingleVideo", path,
                                                                batch_size=self.batch_size,
                                                                dataset=dataset,
                                                                video=video,
                                                                image_size=self.patch_size,
                                                                stride=self.stride,
                                                                n_frames=self.n_frames,
                                                                aug=self.aug,
                                                                dist=self.dist,
                                                                mode=self.mode,
                                                                noise_std=self.noise_std,
                                                                min_noise=self.min_noise,
                                                                max_noise=self.max_noise,
                                                                sample=True)

        sample = self.test_loader.dataset[num][0].unsqueeze(0)[:, :, self.y:self.y + self.h, self.x:self.x + self.w].to(
            self.device)
        return sample

    def get_fixed_noise(self, sample, span=1):
        fixed_noises = []
        for i in range(span + self.n_frames):
            fixed_noises.append(utils.get_noise(sample[:, 0:3, :, :], dist="G", mode='S', noise_std=255))
        return fixed_noises

    def compute_jacobian_filters(self, span=1, load_cache=False, cache_results=False, cache_path=None):
        if load_cache and cache_path is not None and os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=False)

        self.model.eval()

        grad_mapss = []
        pss = []
        qss = []

        for l in tqdm(range(len(self.ps))):
            grad_maps = []
            p = self.ps[l]
            q = self.qs[l]
            pss.append([self.ps[l]])
            qss.append([self.qs[l]])
            for i in range(span):
                sample = self.load_sample(num=self.num + i)
                clean_image = sample[:, (self.mid * self.cpf):((self.mid + 1) * self.cpf), :, :]
                if not self.is_real:
                    noise = (self.noise_std / 255.0) * torch.cat(
                        self.get_fixed_noise(sample, span)[i:i + self.n_frames], 1)
                    noisy_inputs = noise + sample
                    noisy_frame = noisy_inputs[:, (self.mid * self.cpf):((self.mid + 1) * self.cpf), :, :]
                else:
                    noisy_inputs = sample
                    noisy_frame = clean_image
                noisy_inputs = noisy_inputs.requires_grad_(True)

                N, C, H, W = sample.shape
                noise_map = (self.noise_std / 255) * torch.ones(N, 1, H, W).to(self.device)

                if not self.fast:
                    output, _ = self.model(noisy_inputs)

                    if not self.is_real:
                        output, mean_image = utils.post_process(output, noisy_frame, model="blind-video-net",
                                                                sigma=self.noise_std / 255, device=self.device)
                else:
                    output = self.model(noisy_inputs, noise_map)

                loss = 100 * output[:, :, q, p].mean()

                self.model.zero_grad()

                loss.backward()

                grads = noisy_inputs.grad.cpu().detach()

                grad_maps.append(grads)

                img = (np.sum(grads[0, 9:12, :, :].cpu().detach().numpy(), axis=0) / self.cpf).reshape(H, W)

                ptile = 0.5 * np.max(img)
                w_sum = 0
                p_final = 0
                q_final = 0
                for j in range(self.h):
                    for k in range(self.w):
                        if img[j][k] >= ptile:
                            p_final += k * img[j][k]
                            q_final += j * img[j][k]
                            w_sum += img[j][k]
                p_final /= w_sum
                q_final /= w_sum

                pss[l].append(int(p_final))
                qss[l].append(int(q_final))

                p = int(p_final)
                q = int(q_final)

            grad_mapss.append(grad_maps)

        if cache_results:
            if cache_path is None:
                cache_path = "jacobian_filters.pt"
            torch.save((grad_mapss, pss, qss), cache_path)

        return grad_mapss, pss, qss

    def compute_motion_compensation(self, sample, ps, qs, span=1):
        fixed_noises = self.get_fixed_noise(sample, span)
        fpss, fqss, npss, nqss = [], [], [], []
        for l in range(len(ps)):
            fpss.append([ps[l]])
            npss.append([ps[l]])
            fqss.append([qs[l]])
            nqss.append([qs[l]])

        for i in range(span + 1):
            img_0 = np.array(self.to_gray(
                self.test_loader.dataset[self.num + i][0][6:9, self.y:self.y + self.h, self.x:self.x + self.w]))
            img_1 = np.array(self.to_gray(
                self.test_loader.dataset[self.num + i][0][9:12, self.y:self.y + self.h, self.x:self.x + self.w]))
            noisy_img_0 = np.array(self.to_gray(
                self.test_loader.dataset[self.num + i][0][6:9, self.y:self.y + self.h, self.x:self.x + self.w] + (
                        self.noise_std / 255) * fixed_noises[2 + i][0, :, :, :]))
            noisy_img_1 = np.array(self.to_gray(
                self.test_loader.dataset[self.num + i][0][9:12, self.y:self.y + self.h, self.x:self.x + self.w] + (
                        self.noise_std / 255) * fixed_noises[3 + i][0, :, :, :]))

            flow = utils.estimate_invflow(img_0, img_1, "DeepFlow")
            noisy_flow = utils.estimate_invflow(noisy_img_0, noisy_img_1, "DeepFlow")
            for l in range(len(ps)):
                p, q, n_p, n_q = fpss[l][-1], fqss[l][-1], npss[l][-1], nqss[l][-1]
                fpss[l].append(min(p - int(flow[q][p][0]), self.w - 1))
                fqss[l].append(min(q - int(flow[q][p][1]), self.h - 1))
                npss[l].append(min(n_p - int(noisy_flow[n_q][n_p][0]), self.w - 1))
                nqss[l].append(min(n_q - int(noisy_flow[n_q][n_p][1]), self.h - 1))

        return fpss, fqss, npss, nqss

    def plot_motion(self, ps, fpss, fqss, npss, nqss, pss, qss, scale=2, width=2, color="darkorange"):
        head_width = 3 * width
        img = np.array(
            self.transform(self.test_loader.dataset[self.num][0][6:9, self.y:self.y + self.h, self.x:self.x + self.w]))
        noisy_img = np.array(self.transform(
            self.test_loader.dataset[self.num][0][6:9, self.y:self.y + self.h, self.x:self.x + self.w] + (
                    self.noise_std / 255) * self.get_fixed_noise(self.load_sample(), 2)[0][0, :, :, :].cpu()))

        fig, ax = plt.subplots(figsize=(4, 4), frameon=False)

        ax.imshow(noisy_img)
        ax.set_title("Noisy frame")
        ax.axis("off")

        plt.tight_layout()

        fig, ax = plt.subplots(figsize=(4, 4), frameon=False)

        ax.imshow(img)
        ax.set_title("DeepFlow on clean video")
        ax.axis("off")

        for i in range(len(ps)):
            ax.arrow(fpss[i][0], fqss[i][0],
                     scale * (fpss[i][1] - fpss[i][0]),
                     scale * (fqss[i][1] - fqss[i][0]), width=width, head_width=head_width,
                     length_includes_head=False, facecolor=color, edgecolor='midnightblue')

        plt.tight_layout()

        fig, ax = plt.subplots(figsize=(4, 4), frameon=False)

        ax.imshow(img)
        ax.set_title("UDVD implicit motion estimation")
        ax.axis("off")

        for i in range(len(ps)):
            ax.arrow(pss[i][0], qss[i][0],
                     scale * (pss[i][1] - pss[i][0]),
                     scale * (qss[i][1] - qss[i][0]), width=width, head_width=head_width,
                     length_includes_head=False, facecolor=color, edgecolor='midnightblue')

        plt.tight_layout()

        fig, ax = plt.subplots(figsize=(4, 4), frameon=False)

        ax.imshow(img)
        ax.set_title("DeepFlow on noisy video")
        ax.axis("off")

        for i in range(len(ps)):
            ax.arrow(npss[i][0], nqss[i][0],
                     scale * (npss[i][1] - npss[i][0]),
                     scale * (nqss[i][1] - nqss[i][0]), width=width, head_width=head_width,
                     length_includes_head=False, facecolor=color, edgecolor='midnightblue')

        plt.tight_layout()
        plt.show()


def plot_trajectory(ps, trajectory):
    #   deepflow
    fig, ax = plt.subplots(figsize=(4, 4), frameon=False)
    ax.set_title("DeepFlow on noisy video")
    ax.axis("off")
    points = []
    for i in range(len(trajectory)):
        fps, fpss, fqss, npss, nqss, pss, qss = trajectory[i]
        prev_x = fpss[0][0]
        prev_y = fqss[0][0]
        min_x = min(min(fpss[0]), min(npss[0]))
        max_x = max(max(fpss[0]), max(npss[0]))
        min_y = min(min(fqss[0]), min(nqss[0]))
        max_y = max(max(fqss[0]), max(nqss[0]))
        points.append((min_x + prev_x, min_y + prev_y))
        points.append((max_x + prev_x, max_y + prev_y))
        # for j in range(len(ps)):
        # # ax.arrow(fpss[j][0], fqss[j][0],
        # #          2 * (fpss[j][1] - fpss[j][0]),
        # #          2 * (fqss[j][1] - fqss[j][0]), width=2, head_width=6,
        # #          length_includes_head=False, facecolor="darkorange", edgecolor='midnightblue')

    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], color="darkorange")

    plt.tight_layout()
    plt.show()


def construct_trajectory(trajectory_file=None):
    if trajectory_file is None:
        fpss_array, fqss_array, npss_array, nqss_array, pss_array, qss_array = [], [], [], [], [], []
        for i in range(25, 35, 5):
            motion_estimation = MotionEstimation()
            sample = motion_estimation.load_sample(video="01", dataset="KITTI", x=500, y=75, num=i, h=256, w=256)
            grad_mapss, pss, qss = motion_estimation.compute_jacobian_filters(span=1, load_cache=True,
                                                                              cache_path="jacobian_filters.pt")
            ps = motion_estimation.ps[0:9]
            qs = motion_estimation.qs[0:9]
            fpss, fqss, npss, nqss = motion_estimation.compute_motion_compensation(sample, ps, qs, span=1)
            # trajectory.append((ps, fpss, fqss, npss, nqss, pss, qss))
            fpss_array.append(fpss)
            fqss_array.append(fqss)
            npss_array.append(npss)
            nqss_array.append(nqss)
            pss_array.append(pss)
            qss_array.append(qss)
        trajectory = (fpss_array, fqss_array, npss_array, nqss_array, pss_array, qss_array)
        np.save("trajectory.npy", trajectory)
    else:
        trajectory = np.load(trajectory_file)
    plot_trajectory(ps, trajectory)


def main():
    motion_estimation = MotionEstimation()
    # sample = motion_estimation.load_sample(video="01", dataset="KITTI", x=500, y=75, num=50, h=256, w=256)
    sample = motion_estimation.load_sample(video="01", dataset="KITTI", x=500, y=75, num=50, h=256, w=256)
    # grad_mapss, pss, qss = motion_estimation.compute_jacobian_filters(span=1, load_cache=True,
    #                                                                   cache_path="jacobian_filters.pt")
    # grad_mapss, pss, qss = motion_estimation.compute_jacobian_filters(span=1, cache_results=True,
    #                                                                   cache_path="seq1_jacobian_filters.pt")
    grad_mapss, pss, qss = torch.load("seq1_jacobian_filters.pt", weights_only=False)
    ps = motion_estimation.ps[0:9]
    qs = motion_estimation.qs[0:9]
    fpss, fqss, npss, nqss = motion_estimation.compute_motion_compensation(sample, ps, qs, span=1)
    motion_estimation.plot_motion(ps, fpss, fqss, npss, nqss, pss, qss)


if __name__ == "__main__":
    # main()
    construct_trajectory()
