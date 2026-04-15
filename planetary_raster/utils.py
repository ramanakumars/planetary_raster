"""
Reprojection utilities for mapping planetary observation data into arbitrary
output projections.

The core pipeline is:

1. :func:`reproject_image` — inversely transforms output grid coordinates back
   into source pixel space and delegates interpolation to :func:`scatter_to_grid`.
2. :func:`scatter_to_grid` — performs inverse-distance-weighted (IDW)
   nearest-neighbour interpolation to fill each output pixel from the source image.

Helper:

* :func:`extract_segment` — convenience wrapper that builds a local Lambert
  Azimuthal Equal-Area (LAEA) CRS centred on a (lon, lat) point and calls
  :func:`reproject_image`.
* :func:`color_correction` — simple per-channel white-balance normalisation.
"""

import cartopy.crs as ccrs
import numpy as np
from pyproj import crs
from pyproj.transformer import Transformer
from scipy.spatial import cKDTree
from skimage.transform import rescale

from .gridconfig import GridConfig

# Cartopy validates CRS types against a hard-coded allowlist that does not include
# custom pyproj ProjectedCRS objects. This patch adds the missing type strings so
# that Cartopy accepts Jupiter's custom CRS without raising NotImplementedError.
# Can be removed once Cartopy natively supports arbitrary projected CRS — track at
# https://github.com/SciTools/cartopy/issues
ccrs._CRS._expected_types = ("Projected CRS", "Derived Projected CRS")


def reproject_image(
    image_data: np.ndarray,
    projection: crs.coordinate_operation.CoordinateOperation,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    params: GridConfig,
    n_neighbor: int = 5,
    max_dist_neighbors: float = 100e3,
) -> np.ndarray:
    """Reproject a source image onto an output grid defined in a target CRS.

    Uses an inverse-mapping strategy: for each pixel in the output grid the
    corresponding location in the source image is found by transforming the
    output coordinates back through the source CRS, then
    :func:`scatter_to_grid` fills in the pixel values via IDW interpolation.

    :param image_data: Source image array of shape ``(height, width, nchannels)``.
    :type image_data: numpy.ndarray
    :param projection: pyproj CRS of the output grid (e.g. a local LAEA centred
        on the segment of interest).
    :type projection: crs.coordinate_operation.CoordinateOperation
    :param x_grid: X-axis distances for every output pixel, shape
        ``(output_height, output_width)``, units: metres.
    :type x_grid: numpy.ndarray
    :param y_grid: Y-axis distances for every output pixel, same shape as
        ``x_grid``, units: metres.
    :type y_grid: numpy.ndarray
    :param params: Grid configuration carrying the cylindrical CRS and the
        pixel-space KDTree (``pix_tree``) of the source image.
    :type params: GridConfig
    :param n_neighbor: Number of nearest neighbours for IDW interpolation.
        Higher values smooth the output; lower values are faster.
    :type n_neighbor: int
    :param max_dist_neighbors: Maximum distance (metres) within which a source
        neighbour contributes to the interpolated value. Points beyond this
        radius receive zero weight.
    :type max_dist_neighbors: float
    :returns: Reprojected image of shape
        ``(output_height, output_width, nchannels)``.
    :rtype: numpy.ndarray
    """
    # Build a transformer from the output CRS back into Jupiter's cylindrical
    # projection so we can look up source pixel positions for each output pixel.
    inv_transformer = Transformer.from_crs(projection, params.source_crs)

    # Map every output grid point back to cylindrical projected coordinates
    # (metres). Points that fall outside the valid domain become NaN/inf.
    x_t, y_t = inv_transformer.transform(x_grid.flatten(), y_grid.flatten())

    # Stack into (N, 2) array of cylindrical (x, y) positions for each output pixel.
    pix = np.vstack((x_t, y_t)).T

    # Discard output pixels that projected outside the source image coverage.
    pix_masked = pix  # cylindrical coords to look up in source

    return scatter_to_grid(
        image_data.reshape((-1, image_data.shape[-1])),
        pix_masked,
        params.pix_tree,
        x_grid.shape,
        n_neighbor=n_neighbor,
        max_dist_neighbors=max_dist_neighbors,
    )


def scatter_to_grid(
    imgvals: np.ndarray,
    pix: np.ndarray,
    tree: cKDTree,
    img_shape: tuple[int],
    n_neighbor: int,
    max_dist_neighbors: float,
) -> np.ndarray:
    """Interpolate scattered source pixels onto a regular output grid using IDW.

    For each output pixel position given in ``inds``, the ``n_neighbor``
    nearest source pixels are found in ``tree`` and their values are blended
    using inverse-distance weighting. Neighbours further than
    ``max_dist_neighbors`` receive zero weight and do not contribute.

    :param imgvals: Image channel values for every source pixel, shape
        ``(N_src, nchannels)``.
    :type imgvals: numpy.ndarray
    :param pix: Source cylindrical coordinates to look up for each entry in
        ``inds``, shape ``(N_out, 2)``.
    :type pix: numpy.ndarray
    :param tree: KDTree built over the source image's projected pixel
        coordinates, used for nearest-neighbour lookup.
    :type tree: scipy.spatial.cKDTree
    :param img_shape: Shape of the output image ``(height, width)`` or
        ``(N,)`` for a flat output.
    :type img_shape: tuple[int, ...]
    :param n_neighbor: Number of nearest neighbours to blend per output pixel.
    :type n_neighbor: int
    :param max_dist_neighbors: Distance threshold in the same units as the
        KDTree coordinates. Neighbours beyond this distance are zeroed out.
    :type max_dist_neighbors: float
    :returns: Output image of shape ``(*img_shape, nchannels)`` with
        interpolated values at every position in ``inds`` and zeros elsewhere.
    :rtype: numpy.ndarray
    """
    nchannels = imgvals.shape[-1]

    # For each output pixel, find the n_neighbor closest source pixels in
    # projected coordinate space. dist and indi are both (N_out, n_neighbor).
    dist, indi = tree.query(pix, k=n_neighbor)

    # Inverse-distance weights: closer neighbours contribute more.
    # Epsilon (1e-16) avoids division by zero when a query point exactly
    # coincides with a source pixel.
    weight = 1.0 / (dist + 1.0e-16)
    # Zero out contributions from neighbours that are too far away.
    weight[dist > max_dist_neighbors] = 0.0

    weight = weight / np.sum(
        weight, axis=1, keepdims=True
    )  # normalise rows to sum to 1

    # Accumulate weighted channel values for every output pixel.
    newvals = np.zeros((pix.shape[0], nchannels))
    for n in range(nchannels):
        newvals[:, n] = np.sum(np.take(imgvals[:, n], indi, axis=0) * weight, axis=1)

    # Write interpolated values into the output array at the correct 2-D positions.
    output_img = np.zeros((*img_shape, nchannels))
    for n in range(nchannels):
        output_img[..., n] = newvals[:, n].reshape(img_shape)

    output_img[~np.isfinite(output_img)] = 0.0

    return output_img


def extract_segment(
    lon: float,
    lat: float,
    img: np.ndarray,
    params: GridConfig,
    img_size: tuple[int],
    max_dist: float = 8e6,
    n_neighbor: int = 15,
    max_dist_neighbors: float = 500e3,
):
    """Extract a square image patch centred at a given (lon, lat) on Jupiter.

    A Lambert Azimuthal Equal-Area (LAEA) CRS is constructed around the
    requested centre point. This projection preserves area and minimises
    distortion for local patches, making it well-suited for extracting
    spatially consistent cutouts. The source image is then reprojected into
    this local frame via :func:`reproject_image`.

    Returns ``None`` if the reprojected patch contains more than five
    near-zero pixels in the first channel, indicating that the requested
    location falls outside the source image coverage.

    :param lon: Central longitude of the segment in degrees (SysIII).
    :type lon: float
    :param lat: Central latitude of the segment in degrees (planetocentric).
    :type lat: float
    :param img: Full source image array of shape ``(height, width, nchannels)``.
    :type img: numpy.ndarray
    :param params: Grid configuration carrying the Projector and KDTrees.
    :type params: GridConfig
    :param img_size: Output image dimensions as ``(height, width)`` in pixels.
    :type img_size: tuple[int, int]
    :param max_dist: Half-width of the output patch in metres. The output grid
        spans ``[-max_dist, +max_dist]`` in both axes.
    :type max_dist: float
    :param n_neighbor: Number of nearest neighbours for IDW interpolation.
    :type n_neighbor: int
    :param max_dist_neighbors: Maximum source-pixel distance (metres) for a
        neighbour to contribute to the interpolated value.
    :type max_dist_neighbors: float
    :returns: Reprojected image cube of shape ``(height, width, nchannels)``,
        or ``None`` if coverage is insufficient.
    :rtype: numpy.ndarray or None
    """
    # Build a regular metre-space grid centred at the origin of the local LAEA frame.
    x_grid = np.linspace(-max_dist, max_dist, img_size[1])
    y_grid = np.linspace(-max_dist, max_dist, img_size[0])

    XX, YY = np.meshgrid(x_grid, y_grid)

    # Define a Lambert Azimuthal Equal-Area CRS centred on this segment's lon/lat.
    # LAEA is chosen because it preserves area and has low distortion near the
    # projection centre, keeping each extracted patch geometrically consistent.
    laea = crs.coordinate_operation.LambertAzimuthalEqualAreaConversion(
        latitude_natural_origin=lat, longitude_natural_origin=lon
    )
    jupiter_laea = crs.ProjectedCRS(
        laea,
        'Jupiter LAEA',
        crs.coordinate_system.Cartesian2DCS(),
        params.projector.base_crs,
    )

    frames = reproject_image(
        img,
        jupiter_laea,
        XX,
        YY,
        params,
        n_neighbor=n_neighbor,
        max_dist_neighbors=max_dist_neighbors,
    )

    # Reject segments where more than 5 pixels in the first channel are
    # effectively zero — this indicates the patch extends beyond the source
    # image boundary and would produce an incomplete cutout.
    if np.sum(frames[:, :, 0] < 1.0e-10) > 5:
        return None

    # Flip the y-axis: the reprojection produces a top-down array but
    # astronomical images are conventionally stored bottom-up.
    return frames[::-1]


def color_correction(data: np.ndarray) -> np.ndarray:
    """White-balance a multi-channel image by scaling each channel so that the
    brightest pixel in a downsampled version of the image maps to 1.0.

    :param data: image array of shape (height, width, channels), values in [0, 1]
    :returns: white-balanced image normalized to [0, ~1]
    """
    scaled = rescale(data, (0.25, 0.25, 1))
    gray = scaled.mean(axis=-1)
    y, x = np.unravel_index(np.argmax(gray), (gray.shape[0], gray.shape[1]))
    val = 1 / scaled[y, x, :]

    data[:, :, 2] = val[2] * data[:, :, 2]
    data[:, :, 1] = val[1] * data[:, :, 1]
    data[:, :, 0] = val[0] * data[:, :, 0]

    data = data / (1.1 * data.max())

    return data
