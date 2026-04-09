"""Helpers to import an Area geometry from an uploaded file.

Supports geospatial vector files readable by GDAL/OGR (GeoPackage, GeoJSON,
shapefiles, ...) including zipped shapefiles. Multi-feature layers are unioned
into a single (Multi)Polygon, and the geometry can be simplified to keep the
number of vertices manageable.
"""
import os
import shutil
import tempfile
import zipfile

from django.contrib.gis.gdal import DataSource
from django.contrib.gis.geos import GEOSGeometry, MultiPolygon

from dashboard.models import DATA_SRID


def _find_shapefile(directory: str) -> str | None:
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".shp"):
                return os.path.join(root, f)
    return None


def area_file_to_multipolygon(
    file_path: str,
    dest_srid: int = DATA_SRID,
    simplify_tolerance: float = 0.0,
) -> MultiPolygon:
    """Read a vector file and return a GEOS MultiPolygon in ``dest_srid``.

    All polygon features in the first layer are unioned together. If
    ``simplify_tolerance`` is greater than 0, the resulting geometry is
    simplified using the Douglas-Peucker algorithm (units of ``dest_srid`` —
    meters when the project default EPSG:3857 is used).
    """
    cleanup_dir: str | None = None
    try:
        path = file_path
        if zipfile.is_zipfile(file_path):
            cleanup_dir = tempfile.mkdtemp(prefix="area_import_")
            with zipfile.ZipFile(file_path) as z:
                z.extractall(cleanup_dir)
            shp = _find_shapefile(cleanup_dir)
            if shp is None:
                raise ValueError(
                    "Zip archive does not contain a .shp file"
                )
            path = shp

        ds = DataSource(path)
        if ds.layer_count < 1:
            raise ValueError("File contains no layer")
        layer = ds[0]
        if layer.srs is None:
            raise ValueError(
                "The file does not contain a SRS, please provide a file with a SRS"
            )

        polygons: list = []
        for feature in layer:
            geom = feature.geom
            geom.transform(dest_srid)
            geos_geom = GEOSGeometry(geom.wkt, srid=dest_srid)
            if geos_geom.geom_type == "Polygon":
                polygons.append(geos_geom)
            elif geos_geom.geom_type == "MultiPolygon":
                polygons.extend(list(geos_geom))
            else:
                raise ValueError(
                    f"Unsupported geometry type: {geos_geom.geom_type}. "
                    "Only Polygon and MultiPolygon features are supported."
                )

        if not polygons:
            raise ValueError("No polygon features found in the file")

        mp = MultiPolygon(polygons, srid=dest_srid)
        # unary_union dissolves overlaps and adjacent polygons
        result = mp.unary_union

        if simplify_tolerance and simplify_tolerance > 0:
            result = result.simplify(simplify_tolerance, preserve_topology=True)

        if result.geom_type == "Polygon":
            result = MultiPolygon([result], srid=dest_srid)
        elif result.geom_type != "MultiPolygon":
            raise ValueError(
                f"Resulting geometry is not a (Multi)Polygon: {result.geom_type}"
            )
        result.srid = dest_srid
        return result
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
