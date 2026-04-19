import os
import tempfile

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from django.contrib.gis import admin
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.urls import path, reverse
from import_export import resources  # type: ignore
from import_export.admin import ImportExportModelAdmin  # type: ignore
from modeltranslation.admin import TranslationAdmin  # type: ignore

from .import_progress import get_progress, set_progress, clear_progress
from .models import (
    Species,
    Observation,
    DataImport,
    User,
    Dataset,
    BasisOfRecord,
    ObservationComment,
    Area,
    Alert,
    ObservationUnseen,
)

admin.site.site_header = f'{settings.GBIF_ALERT["SITE_NAME"]} administration'


@admin.register(User)
class GbifAlertUserAdmin(DjangoUserAdmin):
    fieldsets = DjangoUserAdmin.fieldsets + (  # type: ignore
        (
            "Custom fields",
            {
                "fields": ("last_visit_news_page", "language"),
            },
        ),
    )


class ObservationCommentCommentInline(admin.TabularInline):
    model = ObservationComment


class ObservationUnseenInline(admin.TabularInline):
    model = ObservationUnseen
    readonly_fields = ["user"]

    # Make that inline read-only
    def has_change_permission(self, request, obj=None):
        return False

    def has_add_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(Observation)
class ObservationAdmin(admin.OSMGeoAdmin):
    list_display = ("stable_id", "date", "species", "source_dataset")
    list_filter = ["data_import", "species"]
    search_fields = ["stable_id"]
    inlines = [ObservationCommentCommentInline, ObservationUnseenInline]


class SpeciesResource(resources.ModelResource):
    # Explicitly list fields so the auto-generated resource doesn't
    # pick up 'tags' (TaggableManager) which can't be imported as a
    # regular field.
    class Meta:
        model = Species
        import_id_fields = ("gbif_taxon_key",)
        exclude = ("id",)
        fields = (
            "name",
            "gbif_taxon_key",
            "vernacular_name",
            "vernacular_name_en",
            "vernacular_name_fr",
            "vernacular_name_nl",
        )

    # camelCase → snake_case header mapping
    COLUMN_MAP = {
        "scientificName": "name",
        "gbifTaxonKey": "gbif_taxon_key",
        "vernacularName": "vernacular_name",
        "vernacularNameEn": "vernacular_name_en",
        "vernacularNameFr": "vernacular_name_fr",
        "vernacularNameNl": "vernacular_name_nl",
    }

    def before_import(self, dataset, using_transactions, dry_run, **kwargs):
        import tablib  # type: ignore

        # Rename camelCase headers to snake_case model field names
        new_headers = [
            self.COLUMN_MAP.get(h, h) for h in dataset.headers
        ]
        # Keep only columns that are in our explicit fields list,
        # and skip duplicates. Work by index to avoid issues with
        # tablib's del-by-name on duplicate headers.
        allowed = set(self.Meta.fields)
        seen = set()
        keep_indices = []
        keep_headers = []
        for i, header in enumerate(new_headers):
            if header in allowed and header not in seen:
                keep_indices.append(i)
                keep_headers.append(header)
                seen.add(header)

        # Build a new dataset with only the kept columns
        new_data = []
        for row in dataset:
            new_data.append(tuple(row[i] for i in keep_indices))
        dataset.wipe()
        dataset.headers = keep_headers
        for row in new_data:
            dataset.append(row)


@admin.register(Species)
class SpeciesAdmin(ImportExportModelAdmin, TranslationAdmin):
    resource_class = SpeciesResource
    list_display = ("name", "vernacular_name", "gbif_taxon_key", "tag_list")
    search_fields = ["name", "vernacular_name", "gbif_taxon_key"]

    def tag_list(self, obj):
        return ", ".join(o.name for o in obj.tags.all())


@admin.register(DataImport)
class DataImportAdmin(admin.ModelAdmin):
    list_display = ("pk", "start", "end", "completed", "imported_observations_counter")
    change_list_template = "admin/dashboard/dataimport/change_list.html"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "trigger-import/",
                self.admin_site.admin_view(self.trigger_import_view),
                name="dashboard_dataimport_trigger_import",
            ),
            path(
                "import-progress/",
                self.admin_site.admin_view(self.import_progress_view),
                name="dashboard_dataimport_import_progress",
            ),
        ]
        return custom_urls + urls

    def trigger_import_view(self, request):
        if request.method != "POST":
            return JsonResponse({"error": "POST required"}, status=405)

        # Check if an import is already running
        progress = get_progress()
        if progress and progress["status"] not in ("completed", "failed"):
            messages.warning(request, "An import is already in progress.")
            return JsonResponse({"error": "Import already in progress"}, status=409)

        # Clear any previous progress and start the job
        clear_progress()
        set_progress("queued", "Import job has been queued, waiting for worker to pick it up...")
        from .views.jobs import run_import_observations

        run_import_observations.delay()
        return JsonResponse({"status": "queued"})

    def import_progress_view(self, request):
        progress = get_progress()
        if progress is None:
            return JsonResponse({"status": "idle", "message": "No import running"})
        return JsonResponse(progress)


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    pass


@admin.register(BasisOfRecord)
class BasisOfRecordAdmin(admin.ModelAdmin):
    pass


@admin.register(Area)
class AreaAdmin(admin.OSMGeoAdmin):
    change_list_template = "admin/dashboard/area/change_list.html"

    def get_queryset(self, request):
        return super().get_queryset(request).prefetch_related("tags")

    def tag_list(self, obj):
        return ", ".join(o.name for o in obj.tags.all())

    list_display = ("name", "owner", "tag_list")

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "import-from-file/",
                self.admin_site.admin_view(self.import_from_file_view),
                name="dashboard_area_import_from_file",
            ),
            path(
                "import-from-file/progress/",
                self.admin_site.admin_view(self.area_import_progress_view),
                name="dashboard_area_import_progress",
            ),
        ]
        return custom_urls + urls

    def import_from_file_view(self, request):
        from .area_import import clear_area_import_progress, get_area_import_progress
        from .forms import AdminAreaImportForm

        if request.method == "POST":
            form = AdminAreaImportForm(request.POST, request.FILES)
            if form.is_valid():
                # Check if an import is already running
                progress = get_area_import_progress()
                if progress and progress["status"] == "running":
                    messages.warning(
                        request, "An area import is already in progress."
                    )
                else:
                    uploaded = request.FILES["data_file"]
                    suffix = os.path.splitext(uploaded.name)[1] or ""
                    # Write to a directory shared between the web and
                    # rqworker containers (they have separate filesystems,
                    # so plain /tmp won't work).
                    shared_dir = os.path.join(
                        settings.BASE_DIR, "shared_tmp"
                    )
                    os.makedirs(shared_dir, exist_ok=True)
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=suffix, dir=shared_dir, delete=False
                    )
                    for chunk in uploaded.chunks():
                        tmp.write(chunk)
                    tmp.close()

                    tolerance = (
                        form.cleaned_data.get("simplify_tolerance") or 0.0
                    )
                    clear_area_import_progress()

                    from .views.jobs import run_area_import

                    run_area_import.delay(
                        file_path=tmp.name,
                        area_name=form.cleaned_data["name"],
                        simplify_tolerance=tolerance,
                    )
                    messages.info(
                        request,
                        "Area import has been queued. "
                        "Progress is shown below.",
                    )
                # Stay on the same page so the user sees the progress bar
        else:
            # Fresh page load — clear any leftover completed/failed status
            # so the user starts with a clean slate.
            progress = get_area_import_progress()
            if progress and progress["status"] in ("completed", "failed"):
                clear_area_import_progress()
            form = AdminAreaImportForm()

        context = {
            **self.admin_site.each_context(request),
            "form": form,
            "opts": self.model._meta,
            "title": "Import area from file",
        }
        return render(
            request, "admin/dashboard/area/import_from_file.html", context
        )

    def area_import_progress_view(self, request):
        from .area_import import get_area_import_progress

        progress = get_area_import_progress()
        if progress is None:
            return JsonResponse({"status": "idle", "message": ""})
        return JsonResponse(progress)


# Beware: the following action is mostly for debugging purposes and will send an email if the usual criteria are not
# met, for example:
# - no unseen observations for the alert
# - the alert is configured for no email notifications
# - a notification e-mail has already been sent recently
@admin.action(description="Send e-mail notifications now for selected alerts")  # type: ignore
def send_alert_notification_email(_modeladmin, _request, queryset):
    for alert in queryset:
        alert.send_notification_email()


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "name",
        "unseen_observations_count",
        "species_list",
        "datasets_list",
        "basis_of_record_list",
        "areas_list",
        "email_notifications_frequency",
    )
    list_filter = ["user", "email_notifications_frequency"]

    actions = [send_alert_notification_email]


@admin.register(ObservationComment)
class ObservationCommentAdmin(admin.ModelAdmin):
    list_display = ("author", "observation")
    list_filter = ["author"]
    raw_id_fields = ("observation",)
