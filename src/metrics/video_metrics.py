import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

class VideoMetrics:
    def __init__(self, device, max_val=1.0, vae=None):
        self.max_val = max_val
        # self.vgg = models.vgg16(pretrained=True).features[:16].eval()
        self.vae = vae
        self._psnr = PeakSignalNoiseRatio().to(device=device)
        self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)

    def psnr(self, frame1, frame2):
        # mse = F.mse_loss(frame1, frame2)
        # psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        psnr = self._psnr(frame1, frame2)
        return psnr.item()

    def ssim(self, frame1, frame2):
        return self._ssim(frame1, frame2).item()

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

    def evaluate_video(self, video1, video2):
        """
        video1 and video2 should be tensors of shape (N, C, H, W)
        where N is the number of frames, C is the number of channels,
        H is the height, and W is the width.
        """
        assert (
            video1.shape == video2.shape
        ), "Input videos must have the same dimensions"

        psnr_values = []
        ssim_values = []

        for i in range(video1.size(0)):
            frame1 = video1[i].unsqueeze(0)  # Add batch dimension
            frame2 = video2[i].unsqueeze(0)  # Add batch dimension

            psnr_values.append(self.psnr(frame1, frame2))
            ssim_values.append(self.ssim(frame1, frame2))
            # latent_l2_values.append(self.latent_l2(frame1, frame2))

        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        # avg_latent_l2 = sum(latent_l2_values) / len(latent_l2_values)

        return avg_psnr, avg_ssim  # , avg_latent_l2
