# pybreaker

`pybreaker` generates synthetic retail transaction line data and then deliberately corrupts it with reusable column-level mutation verbs.

The repo has two main jobs:

1. Generate realistic-enough transaction lines using seeded randomness, item weights, and line math.
2. Apply controlled "breaks" to a pandas `DataFrame` so downstream systems can be tested against bad or missing data.

## Repo structure

- [main.py](E:/pybreaker/main.py) generates a clean transaction dataset and a broken copy.
- [generators/transaction_generator.py](E:/pybreaker/generators/transaction_generator.py) creates synthetic line-level retail transactions.
- [domain/transactions.py](E:/pybreaker/domain/transactions.py) defines the Pydantic and dataclass models used by the generator.
- [breakers/breaker.py](E:/pybreaker/breakers/breaker.py) provides a general mutation engine organized around `verb + data_type`.
- [breakers/detection.py](E:/pybreaker/breakers/detection.py) contains reusable column type-detection and validation logic.

## How generation works

`TransactionGenerator` builds:

- a synthetic store/register structure
- a weighted global item catalog
- line-level transactions with seeded randomness

Each generated row is represented by `RawRetailLine`, which includes:

- payment type
- quantity and unit price in cents
- discount cents
- tax rate in basis points
- tax cents
- extended, net, and line total cents

`RawRetailLine` validates key accounting relationships:

- `extended_cents = quantity * unit_price_cents`
- `net_cents = extended_cents - discount_cents`
- `line_total_cents = net_cents + tax_cents`

## How breakers work

`Breaker` mutates a pandas `DataFrame` by:

- `verb`: the kind of mutation to apply
- `data_type`: the kind of columns to target
- `rate`: the probability that a row in a target column is changed
- `columns`: optional explicit column targeting

If `columns` is omitted, the breaker scans the `DataFrame` and targets every matching column for the requested `data_type`.

If `columns` is provided, the breaker validates that:

- each named column exists
- each named column matches the requested `data_type`

The column detection and validation rules live in a separate detection module so they can be reused independently of the mutation verbs.

### Supported verbs

- `KeyStripper`
  Removes values from UUID columns by replacing a random subset with `pd.NA`.
- `doubler`
  Doubles values in integer, float, or all numeric columns.
- `Nuller`
  Replaces a random subset of values with `pd.NA` for the requested data type.
- `Shifter`
  Nudges numeric values by a small random amount or shuffles selected string values within the same column.
- `DateJitter`
  Adds random offsets to date or datetime columns.
- `scrambler`
  Shuffles characters within selected string values.
- `caser`
  Randomizes lower and upper case within selected string values.
- `scaler`
  Multiplies selected numeric values by a random factor between 0 and 100.

### Supported data types

- `uuid`
- `int`
- `float`
- `numeric`
- `string`
- `date`
- `datetime`

## Example usage

```python
import pandas as pd

from breakers.breaker import Breaker
from generators.transaction_generator import TransactionGenerator

generator = TransactionGenerator(seed=42, store_number=2)
breaker = Breaker(seed=42)

records = [line.model_dump() for line in generator.generate_transactions()]
df_txns = pd.DataFrame(records)

df_broken = breaker.mutate(df_txns, verb="KeyStripper", data_type="uuid", rate=0.08)
df_broken = breaker.mutate(df_broken, verb="Nuller", data_type="string", columns=["payment_type"], rate=0.03)
df_broken = breaker.mutate(df_broken, verb="doubler", data_type="numeric", rate=0.05)
df_broken = breaker.mutate(df_broken, verb="Shifter", data_type="numeric", rate=0.03)
df_broken = breaker.mutate(df_broken, verb="DateJitter", data_type="datetime", columns=["transaction_ts"], rate=0.10)
df_broken = breaker.mutate(df_broken, verb="DateJitter", data_type="date", columns=["business_date"], rate=0.10)
df_broken = breaker.mutate(df_broken, verb="caser", data_type="string", columns=["payment_type"], rate=0.20)
df_broken = breaker.mutate(df_broken, verb="scrambler", data_type="string", columns=["payment_type"], rate=0.02)
df_broken = breaker.mutate(df_broken, verb="scaler", data_type="numeric", columns=["quantity"], rate=0.03)
```

## Current outputs

Running [main.py](E:/pybreaker/main.py) writes:

- `synthetic_transactions.csv`
- `synthetic_transactions_broken_low.csv`
- `synthetic_transactions_broken_medium.csv`
- `synthetic_transactions_broken_high.csv`

The first file is the clean generated dataset. The others are mutated copies at different corruption intensities for testing data quality handling.

Each low, medium, and high scenario applies the full mutation suite, with the
same verbs used in every scenario and only the mutation rates changing by
intensity.

## Notes and future improvements

- The current generator creates timestamps independently for each line, even within the same transaction.
- The breaker API is built in a command-style way around mutation verbs, which makes it easy to add more corruption patterns later.
- Good next additions would be tests for seeded reproducibility and arithmetic invariants.
