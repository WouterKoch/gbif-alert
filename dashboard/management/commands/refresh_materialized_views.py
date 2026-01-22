from django.core.management import BaseCommand

from dashboard.views.helpers import create_or_refresh_all_materialized_views


class Command(BaseCommand):
    help = (
        "Create or refresh all materialized views used for the map hexagon aggregation"
    )

    def handle(self, *args, **options) -> None:
        self.stdout.write("Refreshing all materialized views...")
        create_or_refresh_all_materialized_views()
        self.stdout.write("Done!")
