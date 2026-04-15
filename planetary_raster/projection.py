from dataclasses import dataclass
from enum import Enum

import numpy as np
from pyproj import crs
from pyproj.transformer import Transformer

from .planet import Planet


class InputProjection(str, Enum):
    """Describes the coordinate system of a source observation image.

    Used by :class:`~planetary_raster.gridconfig.GridConfig` to correctly map
    source pixel positions into the source CRS before building the KDTree.
    For a custom pyproj CRS, pass the CRS object directly instead of this enum.
    """
    EQUIRECTANGULAR_PLANETOGRAPHIC = "equirectangular_planetographic"
    EQUIRECTANGULAR_PLANETOCENTRIC = "equirectangular_planetocentric"


def planetocentric_to_planetographic(
    lat_centric_deg: np.ndarray, planet: Planet
) -> np.ndarray:
    """Convert planetocentric latitude to planetographic (geodetic) latitude.

    PyProj's GeographicCRS uses geodetic (planetographic) latitude, so
    source images whose pixels encode planetocentric latitude must be
    converted before building the KDTree.

    Formula: tan(lat_graphic) = tan(lat_centric) / (r_po / r_eq)^2

    :param lat_centric_deg: Planetocentric latitudes in degrees.
    :param planet: Planet whose ellipsoid defines the conversion ratio.
    :returns: Planetographic latitudes in degrees.
    """
    lat_centric = np.deg2rad(lat_centric_deg)
    ratio = (planet.r_po / planet.r_eq) ** 2
    lat_graphic = np.arctan(np.tan(lat_centric) / ratio)
    return np.rad2deg(lat_graphic)


@dataclass
class Bounds:
    """Axis-aligned spatial extent in a given CRS.

    For equirectangular / geographic inputs the units are degrees (SysIII
    longitude, planetographic latitude). For projected inputs the units are
    metres in the relevant CRS.

    :param left: West edge (minimum x / longitude).
    :param right: East edge (maximum x / longitude).
    :param bottom: South edge (minimum y / latitude).
    :param top: North edge (maximum y / latitude).
    """
    left: float
    right: float
    bottom: float
    top: float

    @classmethod
    def from_pyproj_tuple(cls, bounds):
        """Construct from a pyproj ``(west, south, east, north)`` tuple."""
        return cls(left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3])

    @classmethod
    def from_tuple(cls, bounds):
        """Construct from a ``(left, right, bottom, top)`` tuple."""
        return cls(left=bounds[0], right=bounds[1], bottom=bounds[2], top=bounds[3])

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(left, right, bottom, top)``."""
        return [self.left, self.right, self.bottom, self.top]

    def as_proj_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(west, south, east, north)`` for pyproj/rasterio APIs."""
        return [self.left, self.bottom, self.right, self.top]



class BaseProjector:
    """Builds pyproj CRS objects and transformers for a given planet.

    Constructs a geographic CRS (``base_crs``) on the planet's ellipsoid and an
    equidistant cylindrical projection (``eqcyl_projection``) on top of it. These
    are used as the reference metre-space for the KDTree in equirectangular pipelines.

    :param planet: Planet whose ellipsoid and datum define the CRS.

    Attributes set on construction:

    * ``base_crs`` — :class:`pyproj.crs.GeographicCRS` on the planet's ellipsoid.
    * ``eqcyl_projection`` — Equidistant cylindrical :class:`pyproj.crs.ProjectedCRS`.
    * ``transformer`` — Forward transformer: geographic → equidistant cylindrical.
    * ``inv_transformer`` — Inverse transformer: equidistant cylindrical → geographic.
    """

    def __init__(self, planet: Planet):
        self.planet = planet

        # this is the base cylindrical projection for Jupiter
        self.base_crs = crs.GeographicCRS(self.planet.name, datum=self.planet.datum)

        self.eqcyl_projection = crs.ProjectedCRS(
            crs.coordinate_operation.EquidistantCylindricalConversion(),
            f"{planet.name} Cylindrical",
            crs.coordinate_system.Cartesian2DCS(),
            self.base_crs,
        )

        # Store the KDTree in the class
        self.transformer = Transformer.from_crs(self.base_crs, self.eqcyl_projection)
        self.inv_transformer = Transformer.from_crs(
            self.eqcyl_projection, self.base_crs
        )
