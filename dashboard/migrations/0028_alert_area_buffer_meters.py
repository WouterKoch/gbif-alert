from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0027_alert_verified_filter"),
    ]

    operations = [
        migrations.AddField(
            model_name="alert",
            name="area_buffer_meters",
            field=models.PositiveIntegerField(
                default=0,
                help_text=(
                    "Also include observations within this distance (in meters) "
                    "from the selected areas. 0 = strict containment."
                ),
                verbose_name="area buffer (meters)",
            ),
        ),
    ]
