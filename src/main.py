import numpy as np
from segment import segment_3d
from scipy.ndimage import gaussian_filter


def make_synthetic_nuclei(
    shape=(50, 512, 512),
    n_nuclei=100,
    radius_range=(5, 15),
    intensity_range=(0.5, 1.0),
    noise_level=0.1,
    dtype=np.float32,
):
    """Generate synthetic 3D volume with Gaussian blob nuclei."""
    rng = np.random.default_rng()
    volume = (rng.random(shape, dtype=dtype) * noise_level).astype(dtype)  # Background noise

    z_max, y_max, x_max = shape

    for _ in range(n_nuclei):
        # Random position
        z = rng.integers(0, z_max)
        y = rng.integers(0, y_max)
        x = rng.integers(0, x_max)

        # Random radius and intensity
        radius = rng.uniform(*radius_range)
        intensity = rng.uniform(*intensity_range)

        # Create coordinate grids
        zz, yy, xx = np.mgrid[0:z_max, 0:y_max, 0:x_max]

        # Gaussian blob
        dist_sq = (zz - z)**2 + (yy - y)**2 + (xx - x)**2
        blob = intensity * np.exp(-dist_sq / (2 * radius**2))

        # Add to volume
        volume = np.maximum(volume, blob, dtype=dtype)

    # Apply slight blur to make it more realistic
    volume = gaussian_filter(volume, sigma=0.5)

    return volume


# Create synthetic volume with nuclei-like blobs
print("Generating synthetic nuclei volume...")
volume = make_synthetic_nuclei(shape=(50, 512, 512), n_nuclei=100)
print(f"Volume shape: {volume.shape}, range: [{volume.min():.3f}, {volume.max():.3f}]")

print("Invoking 3D segmentation...")
labels, details = segment_3d(volume)
print(f"Detected {labels.max()} nuclei")

# Visualize with ndv
print("Launching ndv viewer...")
import ndv

# Stack volume and labels as separate channels for visualization
# Normalize labels to similar intensity range as volume
labels_normalized = labels.astype(float) / labels.max() if labels.max() > 0 else labels
combined = np.stack([volume, labels_normalized], axis=0, dtype=np.float32)

ndv.imshow(combined, channel_axis=0)
