from abc import ABC
from dataclasses import dataclass

from pyproj import crs


class Planet(ABC):
    """Abstract base class for a planetary body.

    Subclasses must be dataclasses and define :attr:`r_eq`, :attr:`flattening`,
    and :attr:`name`. On instantiation, ``__post_init__`` derives the polar radius
    and constructs the pyproj ellipsoid, prime meridian, and datum objects needed
    to build a geographic CRS for this planet.
    """

    def __post_init__(self):
        """Derive polar radius and build pyproj geodetic objects from r_eq and flattening."""
        self.r_po = self.r_eq * (1 - self.flattening)

        self.ellipsoid = crs.datum.CustomEllipsoid(
            self.name,
            semi_major_axis=self.r_eq,
            semi_minor_axis=self.r_po,
        )
        self.primem = crs.datum.CustomPrimeMeridian(
            longitude=0, name=f'{self.name} Prime Meridian'
        )
        self.datum = crs.datum.CustomDatum(
            self.name, ellipsoid=self.ellipsoid, prime_meridian=self.primem
        )


@dataclass
class Jupiter(Planet):
    """Jupiter's ellipsoid parameters.

    Uses the IAU 2015 reference values: equatorial radius 71,492 km and
    flattening 1/15.41 ≈ 0.06487.
    """
    r_eq: float = 71492e3    # equatorial radius in m
    flattening: float = 0.06487
    name: str = "Jupiter"


#: Registry of supported planets. Keys are lower-case planet names.
#: Pass the key as ``planet_name`` to :class:`~planetary_raster.observation.Observation`.
planets = {"jupiter": Jupiter()}
