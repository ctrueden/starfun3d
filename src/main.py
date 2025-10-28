import numpy as np
from segment import segment_3d
from scipy.ndimage import gaussian_filter


def make_synthetic_nuclei(
    shape=(50, 512, 512),
    n_nuclei=100,
    radius_range=(5, 15),
    intensity_range=(32768, 65535),
    noise_level=6553,
    dtype=np.uint16,
):
    """Generate synthetic 3D volume with Gaussian blob nuclei."""
    rng = np.random.default_rng()
    volume = (rng.random(shape) * noise_level).astype(dtype)  # Background noise

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

        # Add to volume, using float computation then clipping to uint16 range
        volume = np.clip(np.maximum(volume, blob), 0, 65535).astype(dtype)

    # Apply slight blur to make it more realistic (gaussian_filter preserves dtype)
    volume = gaussian_filter(volume.astype(np.float32), sigma=0.5).astype(dtype)

    return volume


# Create synthetic volume with nuclei-like blobs
print("Generating synthetic nuclei volume...")
shape = (256, 256, 256)
volume = make_synthetic_nuclei(shape=shape, n_nuclei=8)
print(f"Volume shape: {volume.shape}, range: [{volume.min():.3f}, {volume.max():.3f}]")

print("Invoking 3D segmentation...")
labels, details = segment_3d(volume)
print(f"Detected {labels.max()} nuclei")

# Visualize with ndv
print("Launching ndv viewer...")
import ndv

# Stack volume and labels as separate channels for visualization
# Normalize labels to similar intensity range as volume
if labels.max() > 0:
    labels_normalized = (labels.astype(np.float32) / labels.max() * 65535).astype(np.uint16)
else:
    labels_normalized = labels.astype(np.uint16)
combined = np.stack([volume, labels_normalized], axis=0)

viewer = ndv.ArrayViewer(combined, channel_axis=0, visible_axes=[1, 2, 3])

#viewer...

viewer.show()
ndv.run_app()
#ndv.imshow(combined, channel_axis=0)
