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
