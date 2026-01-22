# Claude Code Instructions

Project-specific guidance for Claude Code when working on this codebase.

## Code Style

### Formatting
- **Always run `black .` after making changes** to ensure consistent formatting
- The project uses Black for Python code formatting

### Imports
- **Prefer imports at the top of the file** - avoid local/inline imports
- Only use local imports when there's a genuine circular import issue that cannot be resolved otherwise
- Group imports in the standard order: stdlib, third-party, local

## Testing

- Run tests with `pytest` or `python manage.py test`
- The import process tests are in `dashboard/tests/commands/test_import_observations.py`
- Model tests are in `dashboard/tests/models/`

## Key Architecture Notes

### Import Process (`import_observations.py`)
- The observation import is a critical, performance-sensitive process
- It runs inside a database transaction with the site in maintenance mode
- Optimizations are documented in `IMPORT_OBSERVATIONS_OPTIMIZATION.md`
- Avoid N+1 query patterns - batch database operations where possible

### Observation Identity
- Observations are identified across imports by `stable_id` (SHA1 hash of occurrence_id + dataset_key)
- The `replaced_observation` property finds the previous version of an observation
- Comments and unseen status are migrated when observations are replaced

### Seen/Unseen Status
- `ObservationUnseen` tracks which observations users haven't seen yet
- An observation is "unseen" if: it matches a user's alert AND is newer than their notification delay
- The `migrate_unseen_observations()` function updates these during import
