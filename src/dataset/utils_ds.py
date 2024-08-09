import cv2
import numpy as np
import torch
import einops


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def canny(image, use_cuda=False, t_lower=100, t_upper=200, aperture_size=5):
    edge = cv2.Canny(image, t_lower, t_upper, apertureSize=aperture_size)
    return edge


def calculate_normals(depth_map):
    """Calculate surface normals from a depth map."""
    # Assume depth_map is in single-channel format
    depth_map = depth_map.astype(np.float32)

    # Sobel derivatives to find gradients
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=5)

    # Normals calculation
    normals = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float32)
    normals[..., 0] = grad_x
    normals[..., 1] = grad_y
    normals[..., 2] = 1.0  # Z component

    # Normalize to unit vectors
    norms = np.linalg.norm(normals, axis=2)
    normals[..., 0] /= norms
    normals[..., 1] /= norms
    normals[..., 2] /= norms
    return normals


def depth_estimation(image):
    """Placeholder function for depth estimation."""
    # Assume this is a complex function
    depth_map = np.zeros(image.shape[:2])  # Dummy depth map
    return depth_map


class DataProcessor:

    def __init__(
        self, model_type="DPT_Large", t_lower=100, t_upper=200, aperture_size=5
    ):
        self.t_lower = t_lower
        self.t_upper = t_upper
        self.aperture_size = aperture_size
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def apply_canny(self, img):
        image = np.transpose(img, (1, 2, 0))
        img = HWC3(np.uint8(image))
        edge_map2 = cv2.Canny(img, self.t_lower, self.t_upper)
        edge_map = HWC3(edge_map2)
        edge_map_torch = torch.from_numpy(edge_map.copy()).float() / 255.0
        edge_map_torch = einops.rearrange(edge_map_torch, "h w c -> c h w")
        return edge_map_torch

    def estimate_depth(self, img):
        img = cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2RGB)
        input_img = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_img)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu()

    def calculate_normals(self, img):
        # Convert RGB to Grayscale
        image = np.transpose(img, (1, 2, 0))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute gradients along the x and y axis
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)

        # Initialize the normals array
        normals = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.float32)

        # The Z component of the normals is set to a constant value
        normals[..., 0] = grad_x
        normals[..., 1] = grad_y
        normals[..., 2] = 1.0  # Assuming a flat surface in the Z direction

        # Normalize the normal vectors
        norms = np.linalg.norm(normals, axis=2)
        normals[..., 0] /= norms
        normals[..., 1] /= norms
        normals[..., 2] /= norms

        # Scale normals to [0, 255] for visualization
        normals = (normals + 1) / 2 * 255
        normals = normals.astype(np.uint8)
        normals_torch = torch.from_numpy(normals.copy()).float() / 255.0
        normals_torch = einops.rearrange(normals_torch, "h w c -> c h w")
        return normals_torch
