import time
import numpy as np
from segment import segment_3d
from scipy.ndimage import gaussian_filter, zoom


def make_synthetic_nuclei(
    shape=(64, 256, 256),
    n_nuclei=100,
    radius_z_range=(2.5, 4.5),
    radius_xy_range=(16, 22),
    intensity_range=(32768, 65535),
    noise_level=6553,
    dtype=np.uint16,
):
    """
    Generate synthetic 3D volume with anisotropic Gaussian blob nuclei.

    Default radius ranges are centered around model_confocal's training data:
    average nucleus size [x=39, y=39, z=7] pixels. The radius parameter
    represents the standard deviation (σ) of the Gaussian, roughly diameter/2.
    """
    rng = np.random.default_rng()
    volume = (rng.random(shape) * noise_level).astype(dtype)  # Background noise

    z_max, y_max, x_max = shape

    for _ in range(n_nuclei):
        # Random position
        z = rng.integers(0, z_max)
        y = rng.integers(0, y_max)
        x = rng.integers(0, x_max)

        # Random anisotropic radii and intensity
        radius_z = rng.uniform(*radius_z_range)
        radius_xy = rng.uniform(*radius_xy_range)
        intensity = rng.uniform(*intensity_range)

        # Create coordinate grids
        zz, yy, xx = np.mgrid[0:z_max, 0:y_max, 0:x_max]

        # Anisotropic Gaussian blob
        dist_sq = (zz - z)**2 / (2 * radius_z**2) + (yy - y)**2 / (2 * radius_xy**2) + (xx - x)**2 / (2 * radius_xy**2)
        blob = intensity * np.exp(-dist_sq)

        # Add to volume, using float computation then clipping to uint16 range
        volume = np.clip(np.maximum(volume, blob), 0, 65535).astype(dtype)

    # Apply slight blur to make it more realistic (gaussian_filter preserves dtype)
    volume = gaussian_filter(volume.astype(np.float32), sigma=0.5).astype(dtype)

    return volume


# Create synthetic volume with nuclei-like blobs
print("Generating synthetic nuclei volume...")
volume = make_synthetic_nuclei(n_nuclei=8)
print(f"Volume shape: {volume.shape}, range: [{volume.min():.3f}, {volume.max():.3f}]")

print("Invoking 3D segmentation...")
t1 = time.time_ns()
labels, details = segment_3d(volume)
t2 = time.time_ns()
print(f"Detected {labels.max()} nuclei in {(t2-t1)/1_000_000_000} s")

# Visualize with ndv
print("Launching ndv viewer...")
import ndv

# Rescale Z dimension to make visualization more cubic (less squished)
# Target: make Z spacing similar to XY spacing
# With confocal model: avg nucleus [z=7, x=39, y=39], so scale Z by ~39/7 ≈ 5.6
z_scale = 39.0 / 7
zoom_factors = (z_scale, 1, 1)

print(f"Rescaling volume for visualization (Z scale factor: {z_scale})...")
volume_rescaled = zoom(volume, zoom_factors, order=1)
labels_rescaled = zoom(labels, zoom_factors, order=0)  # Nearest neighbor for labels

print(f"Rescaled shape: {volume_rescaled.shape}")

# Stack volume and labels as separate channels for visualization
# Normalize labels to similar intensity range as volume
if labels_rescaled.max() > 0:
    labels_normalized = (labels_rescaled.astype(np.float32) / labels_rescaled.max() * 65535).astype(np.uint16)
else:
    labels_normalized = labels_rescaled.astype(np.uint16)
combined = np.stack([volume_rescaled, labels_normalized], axis=0)
combined = np.stack([volume_rescaled, labels_normalized], axis=0)

viewer = ndv.ArrayViewer(combined, channel_axis=0, visible_axes=[1, 2, 3])

viewer.show()
ndv.run_app()
