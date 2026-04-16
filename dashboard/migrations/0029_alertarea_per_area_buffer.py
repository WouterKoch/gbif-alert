"""Create the AlertArea through model and migrate existing M2M + buffer data."""

from django.db import migrations, models
import django.db.models.deletion


def migrate_alert_areas_forward(apps, schema_editor):
    """Copy rows from the auto M2M table into AlertArea, carrying over
    the per-alert area_buffer_meters value from migration 0028."""
    from django.db import connection

    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO dashboard_alertarea (alert_id, area_id, buffer_meters) "
            "SELECT aa.alert_id, aa.area_id, a.area_buffer_meters "
            "FROM dashboard_alert_areas aa "
            "JOIN dashboard_alert a ON a.id = aa.alert_id"
        )


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0028_alert_area_buffer_meters"),
    ]

    operations = [
        # 1. Create AlertArea table
        migrations.CreateModel(
            name="AlertArea",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "buffer_meters",
                    models.PositiveIntegerField(
                        default=0,
                        help_text="Also include observations within this distance from this area. 0 = strict containment.",
                        verbose_name="buffer (meters)",
                    ),
                ),
                (
                    "alert",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dashboard.alert",
                    ),
                ),
                (
                    "area",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dashboard.area",
                    ),
                ),
            ],
            options={
                "unique_together": {("alert", "area")},
            },
        ),
        # 2. Copy data from old auto M2M table
        migrations.RunPython(
            migrate_alert_areas_forward, migrations.RunPython.noop
        ),
        # 3. Remove old auto M2M and wire up through model
        migrations.RemoveField(model_name="alert", name="areas"),
        migrations.AddField(
            model_name="alert",
            name="areas",
            field=models.ManyToManyField(
                blank=True,
                help_text="Optional (no selection = notify me for all data in the system). To select multiple items, press and hold the Ctrl or Command key and click the items.",
                through="dashboard.AlertArea",
                to="dashboard.area",
                verbose_name="areas",
            ),
        ),
        # 4. Remove old single buffer field
        migrations.RemoveField(model_name="alert", name="area_buffer_meters"),
    ]
