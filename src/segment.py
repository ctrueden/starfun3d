"""3D nuclei segmentation using StarDist."""

import numpy as np
from pathlib import Path
from stardist.models import StarDist3D


def segment_3d(
    image: np.ndarray,
    model_path: str | Path = "models/confocal",
    prob_thresh: float | None = None,
    nms_thresh: float | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Segment nuclei in a 3D volume using a pre-trained StarDist model.

    Parameters
    ----------
    image : np.ndarray
        3D numpy array (Z, Y, X) containing the fluorescence volume to segment.
    model_path : str or Path, optional
        Path to the StarDist model directory containing config.json and weights.
        Default: "models/confocal"
    prob_thresh : float, optional
        Probability threshold for object detection. If None, uses model default.
    nms_thresh : float, optional
        Non-maximum suppression threshold. If None, uses model default.
    normalize : bool, optional
        Whether to normalize the input image. Default: True.

    Returns
    -------
    labels : np.ndarray
        Integer array of same shape as input, where each nucleus has a unique ID.
    details : dict
        Dictionary containing detection details including:
        - 'prob': object probabilities
        - 'dist': distance predictions
        - 'points': detected nucleus centers

    Examples
    --------
    >>> import numpy as np
    >>> from segment import segment_3d
    >>> # Create or load a 3D volume
    >>> volume = np.random.rand(50, 512, 512)  # Example: 50 z-slices
    >>> labels, details = segment_3d(volume)
    >>> print(f"Detected {labels.max()} nuclei")
    """
    # Load the pre-trained model
    model = StarDist3D(None, name="confocal", basedir=Path(model_path).parent)

    # Prepare prediction arguments
    predict_kwargs = {}
    if prob_thresh is not None:
        predict_kwargs["prob_thresh"] = prob_thresh
    if nms_thresh is not None:
        predict_kwargs["nms_thresh"] = nms_thresh

    # Perform segmentation
    labels, details = model.predict_instances(
        image,
        normalize=normalize,
        **predict_kwargs,
    )

    return labels, details


def segment_3d_batch(
    images: list[np.ndarray],
    model_path: str | Path = "models/confocal",
    **kwargs,
) -> list[tuple[np.ndarray, dict]]:
    """
    Segment multiple 3D volumes in batch.

    Parameters
    ----------
    images : list of np.ndarray
        List of 3D numpy arrays to segment.
    model_path : str or Path, optional
        Path to the StarDist model directory.
    **kwargs
        Additional keyword arguments passed to segment_3d.

    Returns
    -------
    results : list of tuple
        List of (labels, details) tuples for each input volume.
    """
    # Load model once for efficiency
    model = StarDist3D(None, name="confocal", basedir=Path(model_path).parent)

    results = []
    for image in images:
        labels, details = model.predict_instances(image, **kwargs)
        results.append((labels, details))

    return results
