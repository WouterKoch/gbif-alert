"""Helpers to import an Area geometry from an uploaded file.

Supports geospatial vector files readable by GDAL/OGR (GeoPackage, GeoJSON,
shapefiles, ...) including zipped shapefiles. Multi-feature layers are unioned
into a single (Multi)Polygon, and the geometry can be simplified to keep the
number of vertices manageable.
"""
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile

import django_rq  # type: ignore
from django.contrib.gis.gdal import DataSource
from django.contrib.gis.geos import GEOSGeometry, MultiPolygon

from dashboard.models import DATA_SRID

logger = logging.getLogger(__name__)

REDIS_KEY = "gbif_alert:area_import_progress"


# ---------------------------------------------------------------------------
# Progress helpers (same pattern as import_progress.py)
# ---------------------------------------------------------------------------


def set_area_import_progress(status: str, message: str) -> None:
    conn = django_rq.get_connection()
    conn.set(
        REDIS_KEY,
        json.dumps(
            {"status": status, "message": message, "timestamp": time.time()}
        ),
    )


def get_area_import_progress() -> dict | None:
    conn = django_rq.get_connection()
    data = conn.get(REDIS_KEY)
    if data:
        return json.loads(data)
    return None


def clear_area_import_progress() -> None:
    conn = django_rq.get_connection()
    conn.delete(REDIS_KEY)


# ---------------------------------------------------------------------------
# Core geometry processing
# ---------------------------------------------------------------------------


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
    progress: bool = False,
) -> MultiPolygon:
    """Read a vector file and return a GEOS MultiPolygon in ``dest_srid``.

    All polygon features in the first layer are unioned together. If
    ``simplify_tolerance`` is greater than 0, the resulting geometry is
    simplified using the Douglas-Peucker algorithm (units of ``dest_srid`` —
    meters when the project default EPSG:3857 is used).

    When *progress* is True, status updates are written to Redis so the admin
    UI can poll them.
    """
    cleanup_dir: str | None = None

    def _report(msg: str) -> None:
        if progress:
            set_area_import_progress("running", msg)
        logger.info("area_file_to_multipolygon: %s", msg)

    try:
        path = file_path
        if zipfile.is_zipfile(file_path):
            _report("Extracting zip archive…")
            cleanup_dir = tempfile.mkdtemp(prefix="area_import_")
            with zipfile.ZipFile(file_path) as z:
                z.extractall(cleanup_dir)
            shp = _find_shapefile(cleanup_dir)
            if shp is None:
                raise ValueError(
                    "Zip archive does not contain a .shp file"
                )
            path = shp

        _report("Reading file…")
        ds = DataSource(path)
        if ds.layer_count < 1:
            raise ValueError("File contains no layer")
        layer = ds[0]
        if layer.srs is None:
            raise ValueError(
                "The file does not contain a SRS, please provide a file with a SRS"
            )

        num_features = layer.num_feat  # type: ignore
        simplify = simplify_tolerance and simplify_tolerance > 0
        _report(
            f"Processing {num_features} feature(s)"
            f"{' (simplifying each to ' + str(simplify_tolerance) + 'm)' if simplify else ''}…"
        )

        polygons: list = []
        for i, feature in enumerate(layer):
            if i % 100 == 0 and i > 0:
                _report(f"Processing feature {i}/{num_features}…")
            geom = feature.geom
            geom.transform(dest_srid)
            # Use WKB for the GDAL → GEOS conversion: it is a compact
            # binary format, whereas WKT is a verbose text representation
            # that can be several times larger in memory.
            geos_geom = GEOSGeometry(memoryview(geom.wkb), srid=dest_srid)

            # Simplify each polygon *before* collecting it.  For large
            # files this is the critical optimisation: detailed vertices
            # are discarded immediately so they never accumulate in the
            # list and never participate in the expensive unary_union.
            if simplify:
                geos_geom = geos_geom.simplify(
                    simplify_tolerance, preserve_topology=True
                )

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

        _report(f"Merging {len(polygons)} polygon(s)…")
        mp = MultiPolygon(polygons, srid=dest_srid)
        # unary_union dissolves overlaps and adjacent polygons
        result = mp.unary_union

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
