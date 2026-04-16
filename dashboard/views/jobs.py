"""Long-running tasks to be used with Django-rq"""
import os

from django.core.management import call_command
from django.db.models import QuerySet
from django_rq import job  # type: ignore

from dashboard.area_import import (
    area_file_to_multipolygon,
    clear_area_import_progress,
    set_area_import_progress,
)
from dashboard.import_progress import set_progress
from dashboard.models import Area, Observation, User


@job
def mark_many_observations_as_seen(observations: QuerySet[Observation], user: User):
    for observation in observations:
        observation.mark_as_seen_by(user)


@job("default", timeout=86400)  # 24 hour timeout
def run_import_observations() -> None:
    """Run the import_observations management command as a background job."""
    try:
        call_command("import_observations")
    except Exception as e:
        set_progress("failed", f"Import failed: {e}")
        raise


@job("default", timeout=3600)  # 1 hour timeout
def run_area_import(
    file_path: str, area_name: str, simplify_tolerance: float
) -> None:
    """Process an uploaded geo file and create an Area in the background."""
    try:
        set_area_import_progress("running", "Starting area import…")
        mpoly = area_file_to_multipolygon(
            file_path,
            simplify_tolerance=simplify_tolerance,
            progress=True,
        )
        set_area_import_progress("running", "Saving area to database…")
        num_vertices = sum(len(p.coords[0]) for p in mpoly)
        num_polygons = len(mpoly)
        Area.objects.create(name=area_name, mpoly=mpoly)
        set_area_import_progress(
            "completed",
            f"Area '{area_name}' imported "
            f"({num_vertices} vertices, {num_polygons} polygon(s)).",
        )
    except Exception as exc:
        set_area_import_progress("failed", f"Import failed: {exc}")
        raise
    finally:
        # Clean up the temp file that was kept alive for the worker
        try:
            os.unlink(file_path)
        except OSError:
            pass
