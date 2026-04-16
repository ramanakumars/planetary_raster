import logging
import time

import numpy as np
from pyproj import crs
from scipy.spatial import cKDTree

from .planet import Planet
from .projection import BaseProjector, Bounds, InputProjection, planetocentric_to_planetographic

logger = logging.getLogger(__name__)


class GridConfig:
    """Maps source image pixels into a spatial KDTree for inverse-mapping reprojection.

    Accepts any input projection — equirectangular (geographic) or an arbitrary
    pyproj ProjectedCRS. Internally resolves the input CRS and builds a KDTree
    in metre space so that ``max_dist_neighbors`` is always expressed in metres.

    :param planet: Planet whose ellipsoid defines the geographic CRS.
    :param bounds: Spatial extent of the source image in ``input_projection`` units.
        For equirectangular inputs: SysIII lon / lat degrees
        (``left`` = lon_start, ``right`` = lon_stop, ``bottom`` = lat_start, ``top`` = lat_stop).
        For projected inputs: extent in that CRS's metres.
    :param image_shape: ``(height, width)`` of the source image in pixels.
    :param input_projection: Coordinate system of the source image pixels.
        Pass an :class:`~bhuvanam.projection.InputProjection` enum value for
        equirectangular sources, or a :class:`pyproj.crs.ProjectedCRS` for any other
        projected source.
    :param filters: Optional list of spectral filter names for this dataset.
    :param segment_size: Half-width of an output segment in metres (used externally).
    """

    def __init__(
        self,
        planet: Planet,
        bounds: Bounds,
        image_shape: tuple[int, int],
        input_projection: InputProjection | crs.ProjectedCRS = (
            InputProjection.EQUIRECTANGULAR_PLANETOGRAPHIC
        ),
        filters: list[str] | None = None,
        segment_size: float = 12e6,
    ):
        self.planet = planet
        self.filters = filters or []
        self.segment_size = segment_size
        self.projector = BaseProjector(planet)

        if isinstance(input_projection, InputProjection):
            self._raw_source_crs = self.projector.base_crs
            self._planetocentric = (
                input_projection == InputProjection.EQUIRECTANGULAR_PLANETOCENTRIC
            )
        elif isinstance(input_projection, crs.ProjectedCRS):
            self._raw_source_crs = input_projection
            self._planetocentric = False
        else:
            raise TypeError(f"Unsupported input_projection type: {type(input_projection)}")

        self._build_grids(bounds, image_shape)

    def _build_grids(self, bounds: Bounds, image_shape: tuple[int, int]):
        """Build pixel coordinate grid and KDTree in source CRS space.

        For geographic (equirectangular) sources the coordinates are projected to
        the planet's equidistant cylindrical CRS so the KDTree is in metres.
        For projected sources the coordinates are used directly.

        Sets:
        * ``x_flat``, ``y_flat`` — source CRS coordinates for every pixel.
        * ``source_crs`` — the pyproj CRS the KDTree is built in.
        * ``pix_tree`` — KDTree over (x_flat, y_flat).
        * ``lonlat_tree``, ``coordinates`` — geographic KDTree (equirectangular only).
        """
        t0 = time.perf_counter()
        cols = np.linspace(bounds.left, bounds.right, image_shape[1])
        # Row 0 in every image format (JPEG, PNG, FITS, GeoTIFF) is the topmost
        # (northernmost) row.  Linspace from top → bottom so that pixel [0, j]
        # maps to bounds.top and pixel [N-1, j] maps to bounds.bottom.
        rows = np.linspace(bounds.top, bounds.bottom, image_shape[0])
        x_src, y_src = np.meshgrid(cols, rows)
        x_flat = x_src.flatten()
        y_flat = y_src.flatten()

        if isinstance(self._raw_source_crs, crs.GeographicCRS):
            # Equirectangular: bounds are SysIII lon/lat degrees.
            # Convert SysIII west-positive → pyproj east-positive longitude.
            lon_flat = 180 - x_flat
            lat_flat = y_flat
            if self._planetocentric:
                lat_flat = planetocentric_to_planetographic(lat_flat, self.planet)
            # Project to eqcyl metres for a uniform, metre-space KDTree.
            self.x_flat, self.y_flat = self.projector.transformer.transform(lon_flat, lat_flat)
            self.source_crs = self.projector.eqcyl_projection
            self.lonlat_tree = cKDTree(np.vstack((lon_flat, lat_flat)).T)
            self.coordinates = np.vstack((lon_flat, lat_flat)).T
        else:
            # Projected CRS: pixel coords are already in metres.
            self.x_flat, self.y_flat = x_flat, y_flat
            self.source_crs = self._raw_source_crs

        self.pix_tree = cKDTree(np.vstack((self.x_flat, self.y_flat)).T)
        n_pixels = image_shape[0] * image_shape[1]
        logger.info(
            "Built source pixel grid (%d×%d = %d pixels) in %.1fs",
            image_shape[1], image_shape[0], n_pixels, time.perf_counter() - t0,
        )

    def __repr__(self) -> str:
        crs_name = getattr(self.source_crs, 'name', str(self.source_crs))
        return f"GridConfig(planet='{self.planet.name}', source_crs='{crs_name}')"
