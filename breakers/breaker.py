from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from breakers.detection import resolve_columns


class Breaker:
    """Apply column-oriented mutations to a pandas DataFrame.

    The public API is organized around `verb + data_type`, for example:

    - `mutate(df, verb="KeyStripper", data_type="uuid")`
    - `mutate(df, verb="doubler", data_type="numeric")`

    If `columns` is omitted, the breaker scans the frame and targets every
    matching column for the requested data type. If `columns` is provided, the
    breaker validates that each named column exists and matches the requested
    type before mutating it.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def mutate(
        self,
        df: pd.DataFrame,
        verb: str,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Dispatch a mutation verb against all matching or explicitly named columns."""
        target_df = df if inplace else df.copy()
        normalized_verb = verb.strip().lower()
        normalized_type = data_type.strip().lower()

        if normalized_verb == "keystripper":
            return self.KeyStripper(
                target_df,
                data_type=normalized_type,
                rate=rate,
                columns=columns,
                inplace=True,
            )

        if normalized_verb == "doubler":
            return self.Doubler(
                target_df,
                data_type=normalized_type,
                rate=rate,
                columns=columns,
                inplace=True,
            )

        if normalized_verb == "nuller":
            return self.Nuller(
                target_df,
                data_type=normalized_type,
                rate=rate,
                columns=columns,
                inplace=True,
            )

        if normalized_verb == "shifter":
            return self.Shifter(
                target_df,
                data_type=normalized_type,
                rate=rate,
                columns=columns,
                inplace=True,
            )

        if normalized_verb == "datejitter":
            return self.DateJitter(
                target_df,
                data_type=normalized_type,
                rate=rate,
                columns=columns,
                inplace=True,
            )

        if normalized_verb == "scrambler":
            return self.Scrambler(
                target_df,
                data_type=normalized_type,
                rate=rate,
                columns=columns,
                inplace=True,
            )

        if normalized_verb == "caser":
            return self.Caser(
                target_df,
                data_type=normalized_type,
                rate=rate,
                columns=columns,
                inplace=True,
            )

        if normalized_verb == "scaler":
            return self.Scaler(
                target_df,
                data_type=normalized_type,
                rate=rate,
                columns=columns,
                inplace=True,
            )

        raise ValueError(f"Unsupported verb: {verb}")

    def KeyStripper(
        self,
        df: pd.DataFrame,
        data_type: str = "uuid",
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Blank out values in UUID columns at a random row-level rate."""
        if data_type != "uuid":
            raise ValueError("KeyStripper currently supports only the 'uuid' data type")

        target_df = df if inplace else df.copy()
        target_columns = self._resolve_columns(target_df, data_type, columns)
        for column in target_columns:
            mask = self._random_mask(len(target_df), rate)
            target_df.loc[mask, column] = pd.NA
        return target_df

    def Doubler(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Multiply targeted numeric values by two."""
        if data_type not in {"int", "float", "numeric"}:
            raise ValueError("doubler supports 'int', 'float', or 'numeric'")

        target_df = df if inplace else df.copy()
        target_columns = self._resolve_columns(target_df, data_type, columns)
        for column in target_columns:
            mask = self._random_mask(len(target_df), rate)
            target_df.loc[mask, column] = target_df.loc[mask, column] * 2
        return target_df

    def Nuller(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Replace targeted values with nulls."""
        target_df = df if inplace else df.copy()
        target_columns = self._resolve_columns(target_df, data_type, columns)
        for column in target_columns:
            mask = self._random_mask(len(target_df), rate)
            target_df.loc[mask, column] = pd.NA
        return target_df

    def Shifter(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Perturb numeric values or reshuffle targeted string values."""
        if data_type not in {"int", "float", "numeric", "string"}:
            raise ValueError("Shifter supports 'int', 'float', 'numeric', or 'string'")

        target_df = df if inplace else df.copy()
        target_columns = self._resolve_columns(target_df, data_type, columns)
        for column in target_columns:
            mask = self._random_mask(len(target_df), rate)
            if data_type in {"int", "numeric"} and is_integer_dtype(target_df[column]):
                deltas = self.rng.integers(-5, 6, size=int(mask.sum()))
                target_df.loc[mask, column] = target_df.loc[mask, column] + deltas
                continue

            if data_type in {"float", "numeric"} and is_float_dtype(target_df[column]):
                deltas = self.rng.uniform(-5.0, 5.0, size=int(mask.sum()))
                target_df.loc[mask, column] = target_df.loc[mask, column] + deltas
                continue

            if data_type == "string":
                values = target_df.loc[mask, column]
                if len(values) > 1:
                    shuffled = values.sample(frac=1.0, random_state=int(self.rng.integers(0, 1_000_000)))
                    target_df.loc[mask, column] = shuffled.to_numpy()
        return target_df

    def DateJitter(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Randomly offset date-like columns by small amounts."""
        if data_type not in {"date", "datetime"}:
            raise ValueError("DateJitter supports 'date' or 'datetime'")

        target_df = df if inplace else df.copy()
        target_columns = self._resolve_columns(target_df, data_type, columns)
        for column in target_columns:
            mask = self._random_mask(len(target_df), rate)
            count = int(mask.sum())
            if count == 0:
                continue

            if data_type == "datetime":
                offsets = pd.to_timedelta(self.rng.integers(-3600, 3601, size=count), unit="s")
                target_df.loc[mask, column] = pd.to_datetime(target_df.loc[mask, column]) + offsets
                continue

            offsets = pd.to_timedelta(self.rng.integers(-2, 3, size=count), unit="D")
            jittered = pd.to_datetime(target_df.loc[mask, column]) + offsets
            target_df.loc[mask, column] = jittered.dt.date
        return target_df

    def Scrambler(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Shuffle characters within targeted string values."""
        if data_type != "string":
            raise ValueError("Scrambler supports only the 'string' data type")

        target_df = df if inplace else df.copy()
        target_columns = self._resolve_columns(target_df, data_type, columns)
        for column in target_columns:
            mask = self._random_mask(len(target_df), rate)
            target_df.loc[mask, column] = target_df.loc[mask, column].map(self._scramble_string_value)
        return target_df

    def Caser(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Randomize upper/lower casing within targeted string values."""
        if data_type != "string":
            raise ValueError("Caser supports only the 'string' data type")

        target_df = df if inplace else df.copy()
        target_columns = self._resolve_columns(target_df, data_type, columns)
        for column in target_columns:
            mask = self._random_mask(len(target_df), rate)
            target_df.loc[mask, column] = target_df.loc[mask, column].map(self._randomize_case_value)
        return target_df

    def Scaler(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Multiply targeted numeric values by a random factor between 0 and 100."""
        if data_type not in {"int", "float", "numeric"}:
            raise ValueError("Scaler supports 'int', 'float', or 'numeric'")

        target_df = df if inplace else df.copy()
        target_columns = self._resolve_columns(target_df, data_type, columns)
        for column in target_columns:
            mask = self._random_mask(len(target_df), rate)
            count = int(mask.sum())
            if count == 0:
                continue

            factors = self.rng.uniform(0.0, 100.0, size=count)
            scaled_values = target_df.loc[mask, column] * factors
            if is_integer_dtype(target_df[column]):
                scaled_values = np.rint(scaled_values).astype("int64")
            target_df.loc[mask, column] = scaled_values
        return target_df

    def _resolve_columns(
        self,
        df: pd.DataFrame,
        data_type: str,
        columns: list[str] | None,
    ) -> list[str]:
        """Resolve target columns using the shared breaker detection module."""
        return resolve_columns(df, data_type, columns)

    def _random_mask(self, size: int, rate: float) -> np.ndarray:
        """Sample a boolean mask with approximately `rate` fraction set to True."""
        if not 0 <= rate <= 1:
            raise ValueError("rate must be between 0 and 1")
        return self.rng.random(size) < rate

    def _scramble_string_value(self, value: object) -> object:
        """Randomly permute characters in a string while leaving nulls untouched."""
        if pd.isna(value):
            return value

        text = str(value)
        if len(text) <= 1:
            return text

        chars = list(text)
        self.rng.shuffle(chars)
        return "".join(chars)

    def _randomize_case_value(self, value: object) -> object:
        """Randomize the casing of alphabetic characters in a string."""
        if pd.isna(value):
            return value

        randomized_chars: list[str] = []
        for char in str(value):
            if char.isalpha():
                randomized_chars.append(char.upper() if self.rng.random() < 0.5 else char.lower())
            else:
                randomized_chars.append(char)
        return "".join(randomized_chars)
