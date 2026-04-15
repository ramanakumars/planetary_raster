import numpy as np
import rasterio
from pyproj import crs
from rasterio.crs import CRS as RioCRS
from rasterio.transform import from_origin

from .projection import Bounds


class Raster:
    """Georeferenced raster with GeoTIFF read/write support.

    Holds a reprojected image array together with the CRS and spatial
    metadata needed to produce a valid GeoTIFF.

    :param data: Image array of shape ``(height, width, channels)``, values in ``[0, 1]``.
    :param projection: pyproj ProjectedCRS of the raster grid.
    :param bounds: Spatial extent in the projection's units (metres).
        ``top`` and ``left`` define the origin of the top-left pixel.
    :param resolution: Pixel size in metres (square pixels assumed).
    """

    def __init__(
        self,
        data: np.ndarray,
        projection: crs.CRS,
        bounds: Bounds,
        resolution: float,
    ):
        self.data = data
        self.projection = projection
        self.bounds = bounds
        self.resolution = resolution
        # Row 0 = north (top-left origin), y pixel size is negative (south).
        self.transform = from_origin(bounds.left, bounds.top, resolution, resolution)

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(height, width, channels)`` of the data array."""
        return self.data.shape

    def to_geotiff(self, path: str) -> None:
        """Write to a GeoTIFF file.

        Data is written as ``float32``. Bands correspond to channels of
        :attr:`data` (e.g. R=1, G=2, B=3 for a 3-channel array).

        :param path: Output file path.
        """
        height, width, n_channels = self.data.shape
        rio_crs = RioCRS.from_wkt(self.projection.to_wkt())

        with rasterio.open(
            path,
            mode='w',
            driver='GTiff',
            height=height,
            width=width,
            count=n_channels,
            dtype='float32',
            crs=rio_crs,
            transform=self.transform,
        ) as dst:
            for i in range(n_channels):
                dst.write(self.data[:, :, i].astype('float32'), i + 1)

    @classmethod
    def from_geotiff(cls, path: str) -> 'Raster':
        """Read a GeoTIFF into a :class:`Raster`.

        :param path: Path to a GeoTIFF file.
        :returns: Reconstructed :class:`Raster` instance.
        """
        with rasterio.open(path) as src:
            data = np.stack([src.read(i + 1) for i in range(src.count)], axis=-1)

            t = src.transform
            bounds = Bounds(
                left=t.c,
                right=t.c + t.a * src.width,
                top=t.f,
                bottom=t.f + t.e * src.height,  # t.e is negative for north-up
            )
            resolution = t.a
            projection = crs.ProjectedCRS.from_wkt(src.crs.to_wkt())

        return cls(data, projection, bounds, resolution)
