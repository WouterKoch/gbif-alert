"""Long-running tasks to be used with Django-rq"""
from django.core.management import call_command
from django.db.models import QuerySet
from django_rq import job  # type: ignore

from dashboard.import_progress import set_progress
from dashboard.models import Observation, User


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
