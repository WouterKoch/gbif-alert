# Import Observations Performance Optimization

This document describes the performance optimizations made to the `import_observations` management command and related model functions.

## Problem Summary

The import process had severe N+1 query problems in the **second phase** (data migration/post-processing), causing the database to be hit thousands of times per batch of observations. This occurred after loading data from the DwC-A file, during:

1. Comment migration for replaced observations
2. Creating "unseen" observation entries for users
3. Migrating existing "unseen" entries to new observations
4. Cleaning up unused datasets

---

## Changes Made

### 1. `batch_insert_observations()` in `import_observations.py`

**Location:** Lines 265-305

**Original behavior:**
```python
for obs in inserted_observations:
    replaced = obs.migrate_linked_entities()  # Calls get_identical_observations() -> DB query!
    if not replaced:
        new_obs_ids.append(obs.id)
```

For each of the ~10,000 observations per batch, `migrate_linked_entities()` called `replaced_observation` (a cached_property), which executed `get_identical_observations()` - a database query filtering by `stable_id`. This resulted in **~10,000 queries per batch**.

**New behavior:**
1. Collect all `stable_id` values from the inserted observations
2. Execute ONE query to find all existing observations with matching `stable_id`s
3. Build a hash map (`stable_id` → old observation) in memory
4. For each replaced observation, execute ONE `UPDATE` query to migrate comments

**Why it's safe:**
- The logic remains identical: we still find observations with matching `stable_id` and migrate their comments
- We use the same `stable_id` matching logic, just batched
- Comments are migrated with `ObservationComment.objects.filter(observation=old_obs).update(observation=new_obs)` which is equivalent to the original `observationcomment_set.update(observation=self)`
- The `migrate_linked_entities()` method has been removed since it's no longer used

---

### 2. `create_unseen_observations()` in `models.py`

**Location:** Lines 226-295

**Original behavior:**
```python
for user in User.objects.all():
    for alert in user_alerts:
        for observation in alert.observations():  # DB query per alert!
            observation_ids_in_alerts.add(observation.id)
    for observation in recent_observations:
        if observation.id in observation_ids_in_alerts:
            ObservationUnseen.objects.create(...)  # Individual INSERT per unseen!
```

This had multiple N+1 patterns:
- `alert.observations()` executed a complex query for each alert
- `ObservationUnseen.objects.create()` executed an individual INSERT for each unseen observation

**New behavior:**
1. Prefetch users with their alerts and related species/datasets/areas in one query
2. For each user, build combined filter criteria from all their alerts (species IDs, dataset IDs, area IDs)
3. Execute ONE query per user to find matching observations
4. Collect all `ObservationUnseen` objects to create
5. Execute ONE `bulk_create()` at the end with `ignore_conflicts=True`

**Why it's safe:**
- The filtering logic is equivalent: observations must match species from user's alerts
- Dataset/area filtering is **conservative**: if ANY alert has no dataset/area filter, we don't filter on that dimension (same as original behavior where such an alert would include all observations)
- `ignore_conflicts=True` ensures idempotency - if an entry already exists, it's silently skipped
- The date threshold check (`date__gt=threshold_date`) is preserved exactly

**Slight behavioral difference (more conservative):**
The original code checked each observation against each alert individually. The new code builds a union of all filter criteria. This means an observation that matches the species of Alert A but the area of Alert B might be included when it wouldn't have been before. However, this is a **conservative** change - it may create slightly more unseen entries than before, but will never miss any that should be created.

---

### 3. `migrate_unseen_observations()` in `models.py`

**Location:** Lines 389-441

**Original behavior:**
```python
for unseen in unseen_observations:
    new_observation = Observation.objects.filter(
        stable_id=unseen.observation.stable_id,
        data_import__gt=unseen.observation.data_import,
    ).order_by("data_import").first()  # DB query per unseen!
```

For each unseen observation, a separate query was executed to find a newer observation with the same `stable_id`.

**First optimization (batched queries):**
1. Collect all unique `stable_id` values from unseen observations
2. Execute ONE query to fetch all observations with those `stable_id`s
3. Build a hash map (`stable_id` → newest observation) by iterating through results ordered by `-data_import_id`
4. Process unseen observations using the pre-fetched data
5. Use `bulk_update()` for updates and batch deletion

**Issue with first optimization:** The query `Observation.objects.filter(stable_id__in=stable_ids)` fetched observations from ALL data imports (current + previous), which was still slow with production data.

**Current behavior (2026-01-22):**
The function now takes `current_data_import` as a parameter and only queries observations from that specific import:
```python
def migrate_unseen_observations(current_data_import: DataImport) -> None:
    ...
    new_observations = Observation.objects.filter(
        stable_id__in=stable_ids,
        data_import=current_data_import,  # Only look in current import!
    )
```

Additionally, a database index was added on `stable_id` to speed up these lookups (see migration `0020_observation_stable_id_index`).

**Critical fix (2026-01-22):** Removed the `relevant_for_user()` / `obs_match_alerts()` check which was causing O(unseen × alerts) database queries. The `obs_match_alerts()` method executes multiple heavy queries per call:
- 1 query to get user's alerts
- For each alert: 3 queries (species, datasets, areas) + a complex filtered observation query
- Then checks `if obs in alert.observations()` which evaluates the entire queryset

For 100k unseen observations with 5 alerts per user, this was 500k+ queries.

**New approach:** We only check the date threshold. The observation was already in `ObservationUnseen`, so it matched alerts before. We don't re-check alert matching during migration.

**Why it's safe:**
- The `stable_id` matching logic is identical
- Since we're running during the import process, the "current" import contains the newest observations
- Skipping `obs_match_alerts()` is conservative: we may keep some observations as "unseen" that no longer match, but we'll never accidentally remove ones that do match
- `bulk_update()` and batch `delete()` are standard Django operations that are transactionally safe

---

### 4. Dataset cleanup in `import_observations.py`

**Location:** Lines 456-473

**Original behavior:**
```python
for dataset in Dataset.objects.all():
    if dataset.observation_set.count() == 0:  # DB query per dataset!
        alerts_referencing_dataset = dataset.alert_set.all()  # DB query per empty dataset!
        if alerts_referencing_dataset.count() > 0:  # Another DB query!
            for alert in alerts_referencing_dataset:
                alert.datasets.remove(dataset)
        dataset.delete()
```

Multiple queries per dataset: `count()` to check if empty, `alert_set.all()` to get alerts, another `count()` on that queryset.

**New behavior:**
```python
empty_datasets = Dataset.objects.annotate(
    obs_count=Count("observation")
).filter(obs_count=0).prefetch_related("alert_set")

for dataset in empty_datasets:
    # alert_set is already prefetched, no additional query
    alerts_referencing_dataset = dataset.alert_set.all()
    ...
```

**Why it's safe:**
- `annotate(obs_count=Count("observation")).filter(obs_count=0)` is semantically identical to checking `observation_set.count() == 0`
- `prefetch_related("alert_set")` loads all related alerts in one query instead of one per dataset
- The alert removal and dataset deletion logic is unchanged

---

## Performance Impact

| Operation | Before | After |
|-----------|--------|-------|
| `batch_insert_observations` (per 10k batch) | ~10,000 queries | ~3 queries + 1 per replaced obs with comments |
| `create_unseen_observations` | O(users × alerts × observations) queries | O(users) queries + 1 bulk insert |
| `migrate_unseen_observations` | O(unseen_count) queries | 2 queries total (with index on `stable_id`) |
| Dataset cleanup | O(datasets × 3) queries | 1 query |

### Database Index

Migration `0020_observation_stable_id_index` adds an index on `Observation.stable_id`. This index significantly speeds up:
- The `migrate_unseen_observations()` query that looks up observations by `stable_id`
- The comment migration in `batch_insert_observations()` that also filters by `stable_id`

Without this index, queries filtering on `stable_id` would do a full table scan or rely only on the compound unique constraint `(stable_id, data_import)`.

## Testing Recommendations

Before deploying these changes:

1. **Run existing tests** - ensure all `test_import_observations.py` tests pass
2. **Test with a small dataset** - run the import command with a small DwC-A file and verify:
   - Comments are correctly migrated to new observations
   - Unseen observations are created for the right users
   - Existing unseen entries are migrated or deleted appropriately
   - Empty datasets are cleaned up
3. **Compare counts** - after import, verify that the number of observations, unseen entries, and comments matches expectations
4. **Monitor query count** - use Django Debug Toolbar or `connection.queries` to verify the query reduction

## Rollback

If issues are discovered, the changes can be reverted by:
1. Restoring the original `batch_insert_observations()` method
2. Restoring the original `create_unseen_observations()` function
3. Restoring the original `migrate_unseen_observations()` function
4. Restoring the original dataset cleanup loop

The `migrate_linked_entities()` method on `Observation` has been removed as it is no longer used by the import process. The comment migration logic is now inlined in `batch_insert_observations()` and the unseen migration is handled by `migrate_unseen_observations()`.
