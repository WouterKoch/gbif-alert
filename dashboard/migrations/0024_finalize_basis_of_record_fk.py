from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("dashboard", "0023_populate_basis_of_record_fk"),
    ]

    operations = [
        # 1. Remove the old TextField
        migrations.RemoveField(
            model_name="observation",
            name="basis_of_record",
        ),
        # 2. Rename the FK from basis_of_record_fk to basis_of_record
        migrations.RenameField(
            model_name="observation",
            old_name="basis_of_record_fk",
            new_name="basis_of_record",
        ),
        # 3. Make it non-nullable
        migrations.AlterField(
            model_name="observation",
            name="basis_of_record",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                to="dashboard.basisofrecord",
            ),
        ),
    ]
