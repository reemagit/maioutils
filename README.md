# maioutils

`maioutils` is a small collection of reusable research, data-analysis, and
general Python utilities. Its first utility generates synthetic pandas
DataFrames that resemble the schema and broad statistical characteristics of
source data.

The generator is intended for development and testing. It is not a formal
privacy mechanism and does not provide differential privacy guarantees.

## Installation

From a local clone, install the package with:

```bash
pip install .
```

For editable development and testing:

```bash
pip install -e ".[dev]"
```

## Example

```python
from maioutils.synthetic import make_synthetic_dataframe

fake_df = make_synthetic_dataframe(
    df,
    id_in_index=True,
    keep_ids=False,
    random_state=42,
)
```

Additional utilities may be added later as focused subpackages such as
`validation`, `stats`, or `plotting`.

## Publishing changes

After editing the package locally, run:

```bash
./scripts/publish.sh "Describe the change"
```

The script runs the test suite, displays the files that will be committed,
asks for confirmation, commits the changes, incorporates any upstream changes,
and pushes the current branch to GitHub. Preview it without changing Git or
GitHub with:

```bash
./scripts/publish.sh --dry-run
```

Versions are unchanged by default. Optionally increment the semantic version
stored in `pyproject.toml` as part of the same commit:

```bash
./scripts/publish.sh --bump patch "Fix synthetic ID generation"
./scripts/publish.sh --bump minor "Add a new utility"
./scripts/publish.sh --bump major "Change the public API"
```

Use `--dry-run` together with `--bump` to preview the next version without
editing any files:

```bash
./scripts/publish.sh --dry-run --bump patch
```

## Updating a remote installation

On a remote machine that has cloned the repository, activate the environment
where `maioutils` is installed and run:

```bash
cd /path/to/maioutils
source /path/to/venv/bin/activate
./scripts/update_remote.sh
```

Active Conda environments are detected automatically through `CONDA_PREFIX`.
If no virtualenv or Conda environment is active, the updater uses `python` or
`python3` from the current shell.

Alternatively, pass the environment or Python executable directly:

```bash
./scripts/update_remote.sh /path/to/venv
```

The updater refuses to overwrite local server changes, pulls the current
branch using fast-forward only, and reinstalls the package in non-editable
mode. Restart any process that already imported `maioutils` afterward.
