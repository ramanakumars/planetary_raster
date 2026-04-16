# bhuvanam 

Reproject planetary observation images into any map projection using pyproj.
Source images can be equirectangular mosaics (FITS, JPEG, PNG) or previously
reprojected GeoTIFFs. Output is a georeferenced `Raster` that can be written
to a GeoTIFF and loaded into any GIS tool.

Bhuvanam (bhoo-vuh-num) is the Sanskrit word describing either the universe, Earth or the planets

## Install

```bash
uv sync          # installs all dependencies from pyproject.toml
```

## Core concepts

| Class | Role |
|---|---|
| `Observation` | Loads a source image and builds the spatial index for reprojection |
| `Raster` | Georeferenced output array — read/write GeoTIFF |
| `GridConfig` | Maps source pixels to a KDTree in the source CRS (built by `Observation`) |
| `Bounds` | Axis-aligned spatial extent `(left, right, bottom, top)` in CRS units |
| `InputProjection` | Enum describing the source pixel coordinate convention |

### Reprojection pipeline

```
Observation.project_to(target_crs, bounds, resolution)
    │
    ├─ for each output pixel: transform target_crs → source_crs
    ├─ KDTree lookup → nearest source pixels
    ├─ IDW blend → output pixel value
    │
    └─ Raster(data, target_crs, bounds, resolution)
         row 0 = northernmost row (GeoTIFF north-up convention)
```

## Coordinate conventions

### Longitude — SysIII west-positive vs. pyproj east-positive

`Bounds` for equirectangular source images uses **SysIII west-positive** longitude.
Internally this is converted to pyproj east-positive via `lon_east = 180 − lon_sysIII`
before the KDTree is built.

When constructing a **target CRS** (e.g. LAEA), `longitude_natural_origin` must be
in **east-positive** degrees:

```
SysIII 340° W  →  180 − 340 = −160° E
SysIII 200° W  →  180 − 200 = −20°  E
```

### Bounds orientation

`Bounds.bottom` must be ≤ `Bounds.top`. For a full-disk equirectangular image
use `Bounds(left=360, right=0, bottom=-90, top=90)`. Passing inverted latitude
raises `ValueError`. For JPEG and PNG images where the convention is north-down,
this is handled internally in the code.

### Output orientation

All output from `project_to` follows the **GeoTIFF north-up** convention:
`raster.data[0, :, :]` is the northernmost row. Display with matplotlib's default
`ax.imshow(raster.data)` (no `origin` argument needed).

## Example 1 — FITS equirectangular mosaic

A full-disk Jupiter mosaic in FITS format, stored as an equirectangular
(plate carrée) grid in System III west longitude / planetographic latitude.

```python
from pyproj import crs
from bhuvanam.observation import Observation
from bhuvanam.projection import Bounds, InputProjection

# Load source observation
# bounds: SysIII West lon 0–360°, planetographic lat −90–90°
obs = Observation(
    "jupiter_mosaic.fits",
    planet="jupiter",
    bounds=Bounds(left=360, right=0, bottom=-90, top=90),
    input_projection=InputProjection.EQUIRECTANGULAR_PLANETOGRAPHIC,
)

# obs.base_crs is the planet's geographic CRS — use it as the geodetic base
# for any target projection.
laea_op = crs.coordinate_operation.LambertAzimuthalEqualAreaConversion(
    latitude_natural_origin=20,    # degrees, planetographic
    longitude_natural_origin=-160, # east-positive: 180 - SysIII_lon
                                   # e.g. SysIII 340° → 180 - 340 = -160°
)
laea_crs = crs.ProjectedCRS(
    laea_op,
    "Jupiter LAEA",
    crs.coordinate_system.Cartesian2DCS(),
    obs.base_crs,
)

# Reproject a 4000 km × 4000 km patch at 10 km/pixel
raster = obs.project_to(
    projection=laea_crs,
    bounds=Bounds(left=-2e6, right=2e6, bottom=-2e6, top=2e6),
    resolution=10e3,
)

print(raster)            # Raster(shape=(400, 400, N), resolution=10000m, crs='Jupiter LAEA')
print(raster.bounds)     # actual output extent in metres
raster.to_geotiff("patch.tif")
```

### Planetocentric latitude source

Some FITS files store latitudes in **planetocentric** convention rather than
planetographic. Pass `EQUIRECTANGULAR_PLANETOCENTRIC` to apply the correct
conversion before building the KDTree (difference reaches ~3.8° at ±45° for
Jupiter):

```python
obs = Observation(
    "jupiter_mosaic_centric.fits",
    planet="jupiter",
    bounds=Bounds(left=360, right=0, bottom=-90, top=90),
    input_projection=InputProjection.EQUIRECTANGULAR_PLANETOCENTRIC,
)
```

## Example 2 — Reprojecting a GeoTIFF

A GeoTIFF written by a previous `raster.to_geotiff()` call can be loaded back
as an `Observation` for further reprojection. The CRS and bounds are read from
the file's geospatial metadata — no additional arguments needed.

```python
from bhuvanam.observation import Observation
from bhuvanam.projection import Bounds
from pyproj import crs

# Load a previously reprojected GeoTIFF
obs = Observation("patch.tif", planet="jupiter")

# Reproject into a different projection (e.g. equidistant cylindrical)
eqcyl_crs = obs.gridconfig.projector.eqcyl_projection

raster = obs.project_to(
    projection=eqcyl_crs,
    bounds=Bounds(left=-5e6, right=5e6, bottom=-3e6, top=3e6),
    resolution=5e3,
)

raster.to_geotiff("patch_cylindrical.tif")
```

### Reading a GeoTIFF directly

`Raster.from_geotiff` reconstructs the full object including bounds and CRS:

```python
from bhuvanam.raster import Raster

raster = Raster.from_geotiff("patch.tif")
print(raster.shape)       # (height, width, channels)
print(raster.bounds)      # Bounds in the output CRS's units
print(raster.resolution)  # pixel size in metres
print(raster.projection)  # pyproj CRS
```

## Supported source formats

| Format | Extension | Notes |
|---|---|---|
| FITS | `.fits` | Values used as-is (no normalisation) |
| JPEG | `.jpg`, `.jpeg` | Pixel values normalised to `[0, 1]` |
| PNG | `.png` | Pixel values used as-is (already `[0, 1]` for float PNG) |
| GeoTIFF | `.tif`, `.tiff` | CRS and bounds read from file metadata — `bounds` and `input_projection` not required |

> `.tif` / `.tiff` files are always treated as GeoTIFFs. Pass JPEG or PNG for
> non-georeferenced equirectangular mosaics.

## Using a custom planet

Subclass `Planet` as a dataclass and pass an instance directly to `Observation`:

```python
from dataclasses import dataclass
from bhuvanam.planet import Planet
from bhuvanam.observation import Observation
from bhuvanam.projection import Bounds, InputProjection

@dataclass
class Saturn(Planet):
    r_eq: float = 60268e3
    flattening: float = 0.09796
    name: str = "Saturn"

obs = Observation(
    "saturn_mosaic.fits",
    planet=Saturn(),
    bounds=Bounds(left=360, right=0, bottom=-90, top=90),
    input_projection=InputProjection.EQUIRECTANGULAR_PLANETOGRAPHIC,
)
```

## Progress logging

KDTree construction is logged at `INFO` level. Enable it with:

```python
import logging
logging.basicConfig(level=logging.INFO)
```
