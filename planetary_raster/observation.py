from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from pyproj import crs

from .gridconfig import GridConfig
from .planet import Planet, planets
from .projection import Bounds, InputProjection
from .raster import Raster
from .utils import reproject_image


class Observation:
    """A source planetary image ready for reprojection.

    Loads an image file and builds the spatial index (:class:`GridConfig`)
    that maps source pixels to geographic coordinates, enabling reprojection
    to any target CRS via :meth:`project_to`.

    :param file_path: Path to the source image (.fits, .tif, .jpg, .png).
    :param planet: A :class:`~planetary_raster.planet.Planet` instance, or a
        string key from the :data:`~planetary_raster.planet.planets` registry
        (e.g. ``"jupiter"``). Pass a custom ``Planet`` subclass instance to use
        a body not in the registry.
    :param bounds: Spatial extent of the source image in ``input_projection`` units.
        For equirectangular inputs: SysIII lon/lat degrees.
        For projected inputs: metres in the given CRS.
    :param input_projection: Coordinate system of the source image pixels.
    """

    def __init__(
        self,
        file_path: str,
        planet: Planet | str,
        bounds: Bounds | None = None,
        input_projection: InputProjection | crs.ProjectedCRS | None = None,
    ):
        self.planet = planets[planet] if isinstance(planet, str) else planet

        # try to load the data from GeoTiff
        if Path(file_path).suffix.lower() in (".tif", ".tiff"):
            self.load_geotiff(file_path)
            return

        # all other data that is not GeoTiff
        if bounds is None or input_projection is None:
            raise ValueError(
                "CRS and bounds information is required for non-GeoTIFF images"
            )

        self.load_image(file_path, bounds, input_projection)

    def load_geotiff(self, file_path: str):
        """Load a GeoTIFF written by :meth:`Raster.to_geotiff`.

        The CRS and bounds are read from the file's geospatial metadata so no
        additional arguments are needed.  The projection stored in the file is
        passed directly to :class:`~planetary_raster.gridconfig.GridConfig`.

        :param file_path: Path to a GeoTIFF file.
        """
        raster = Raster.from_geotiff(file_path)
        self.image = raster.data
        self.gridconfig = GridConfig(
            planet=self.planet,
            bounds=raster.bounds,
            image_shape=(self.image.shape[0], self.image.shape[1]),
            input_projection=raster.projection,
        )

    def load_image(
        self,
        file_path: str,
        bounds: Bounds,
        input_projection: InputProjection | crs.ProjectedCRS,
    ):
        """Load a non-GeoTIFF image and build the spatial index.

        Reads pixel data from ``file_path`` and constructs a
        :class:`~planetary_raster.gridconfig.GridConfig` from the supplied
        bounds and projection metadata.

        :param file_path: Path to the image (.fits, .jpg, .jpeg, .png).
        :param bounds: Spatial extent in ``input_projection`` units.
        :param input_projection: Coordinate system of the source pixels.
        """
        suffix = Path(file_path).suffix.lower()
        if suffix == '.fits':
            with fits.open(file_path) as inhdu:
                self.image = np.atleast_3d(inhdu[0].data[:])
        elif suffix in ('.tif', '.tiff', '.jpg', '.jpeg'):
            self.image = np.atleast_3d(plt.imread(file_path)) / 255.0
        elif suffix in ('.png'):
            self.image = np.atleast_3d(plt.imread(file_path))
        else:
            raise ValueError(
                f"Unsupported input format '{suffix}'. "
                "Expected .fits, .tif/.tiff, .jpg/.jpeg, or .png"
            )
        self.gridconfig = GridConfig(
            planet=self.planet,
            bounds=bounds,
            image_shape=(self.image.shape[0], self.image.shape[1]),
            input_projection=input_projection,
        )

    def project_to(
        self,
        projection: crs.ProjectedCRS,
        bounds: Bounds,
        resolution: float,
        n_neighbor: int = 10,
        max_dist_neighbors: float = 500e3,
    ) -> Raster:
        """Reproject the observation into a target CRS and return a :class:`Raster`.

        :param projection: Target pyproj ProjectedCRS.
        :param bounds: Output extent in target CRS metres.
        :param resolution: Output pixel size in metres.
        :param n_neighbor: Number of nearest-neighbour source pixels for IDW blending.
        :param max_dist_neighbors: Max source distance (metres) for a neighbour to
            contribute.
        :returns: Georeferenced reprojected image.
        """
        x_grid = np.arange(bounds.left, bounds.right, resolution)
        # North-down: row 0 = top (northernmost), consistent with GeoTIFF convention.
        y_grid = np.arange(bounds.bottom, bounds.top, resolution)

        XX, YY = np.meshgrid(x_grid, y_grid)

        data = reproject_image(
            self.image,
            projection,
            XX,
            YY,
            self.gridconfig,
            n_neighbor=n_neighbor,
            max_dist_neighbors=max_dist_neighbors,
        )

        # Compute actual bounds from the generated grid (arange may stop before right/bottom)
        actual_bounds = Bounds(
            left=bounds.left,
            right=bounds.left + data.shape[1] * resolution,
            top=bounds.top,
            bottom=bounds.top - data.shape[0] * resolution,
        )

        return Raster(data, projection, actual_bounds, resolution)
