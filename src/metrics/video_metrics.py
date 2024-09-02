import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from einops import rearrange
import torch


class VideoMetrics:
    def __init__(self, device, max_val=1.0, vae=None):
        self.max_val = max_val
        # self.vgg = models.vgg16(pretrained=True).features[:16].eval()
        self.vae = vae
        self._psnr = PeakSignalNoiseRatio((-1, 1)).to(device=device)
        self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)
        self._fid = FrechetInceptionDistance().to(device=device)

    def psnr(self, frame1, frame2):
        psnr = self._psnr(frame1, frame2)
        return psnr.item()

    def ssim(self, frame1, frame2):
        return self._ssim(frame1, frame2).item()

    def fid(self, frame1, frame2):
        frame1 = (((frame1 + 1) * 0.5) * 255).to(torch.uint8)
        frame2 = (frame2 * 255).to(torch.uint8)
        self._fid.update(frame2, real=True)
        self._fid.update(frame1, real=False)
        result = self._fid.compute()
        return result

    def latent_l2(self, frame1, frame2):
        if self.vae is not None:
            frame1_features = self.vgg(frame1)
            frame2_features = self.vgg(frame2)
            latent_l2_dist = F.mse_loss(
                frame1_features, frame2_features, reduction="mean"
            )
            return latent_l2_dist.item()
        else:
            return 0.0

    def evaluate_video(self, video1, video2, mask_num=2):
        """
        video1 and video2 should be tensors of shape (N, C, H, W)
        where N is the number of frames, C is the number of channels,
        H is the height, and W is the width.
        """
        assert (
            video1.shape == video2.shape
        ), "Input videos must have the same dimensions"
        video_pred = rearrange(
            video1[:, mask_num:, :, :, :], "b f c h w -> (b f) c h w"
        )
        video_true = rearrange(
            video2[:, mask_num:, :, :, :], "b f c h w -> (b f) c h w"
        )

        psnr_values = []
        ssim_values = []
        # fid_values = []

        for i in range(video1.size(0)):
            frame1 = video_pred[i].unsqueeze(0)  # Add batch dimension
            frame2 = video_true[i].unsqueeze(0)  # Add batch dimension

            psnr_values.append(self.psnr(video_pred[i], video_true[i]))
            ssim_values.append(self.ssim(frame1, frame2))

        avg_fid = self.fid(video_pred, video_true)
        # latent_l2_values.append(self.latent_l2(frame1, frame2))

        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        # avg_fid = sum(fid_values) / len(fid_values)
        # avg_latent_l2 = sum(latent_l2_values) / len(latent_l2_values)

        return avg_psnr, avg_ssim, avg_fid
