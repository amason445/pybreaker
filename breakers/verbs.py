from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from breakers.detection import resolve_columns


class BreakerVerb(ABC):
    """Base class for breaker verbs in a factory-style mutation system."""

    supported_data_types: set[str] | None = None

    def apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Validate type support and apply a mutation to the selected DataFrame."""
        normalized_type = data_type.strip().lower()
        if self.supported_data_types is not None and normalized_type not in self.supported_data_types:
            supported = ", ".join(sorted(self.supported_data_types))
            raise ValueError(f"{self.__class__.__name__} supports {supported}")

        target_df = df if inplace else df.copy()
        target_columns = resolve_columns(target_df, normalized_type, columns)
        return self._apply(target_df, rng, normalized_type, rate, target_columns)

    @abstractmethod
    def _apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        data_type: str,
        rate: float,
        columns: list[str],
    ) -> pd.DataFrame:
        """Apply a concrete mutation implementation."""

    def _random_mask(self, rng: np.random.Generator, size: int, rate: float) -> np.ndarray:
        if not 0 <= rate <= 1:
            raise ValueError("rate must be between 0 and 1")
        return rng.random(size) < rate

    def _scramble_string_value(self, rng: np.random.Generator, value: object) -> object:
        if pd.isna(value):
            return value

        text = str(value)
        if len(text) <= 1:
            return text

        chars = np.array(list(text), dtype=object)
        rng.shuffle(chars)
        return "".join(chars.tolist())

    def _randomize_case_value(self, rng: np.random.Generator, value: object) -> object:
        if pd.isna(value):
            return value

        randomized_chars: list[str] = []
        for char in str(value):
            if char.isalpha():
                randomized_chars.append(char.upper() if rng.random() < 0.5 else char.lower())
            else:
                randomized_chars.append(char)
        return "".join(randomized_chars)


class KeyStripperVerb(BreakerVerb):
    supported_data_types = {"uuid"}

    def _apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        data_type: str,
        rate: float,
        columns: list[str],
    ) -> pd.DataFrame:
        for column in columns:
            mask = self._random_mask(rng, len(df), rate)
            df.loc[mask, column] = pd.NA
        return df


class DoublerVerb(BreakerVerb):
    supported_data_types = {"int", "float", "numeric"}

    def _apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        data_type: str,
        rate: float,
        columns: list[str],
    ) -> pd.DataFrame:
        for column in columns:
            mask = self._random_mask(rng, len(df), rate)
            df.loc[mask, column] = df.loc[mask, column] * 2
        return df


class NullerVerb(BreakerVerb):
    def _apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        data_type: str,
        rate: float,
        columns: list[str],
    ) -> pd.DataFrame:
        for column in columns:
            mask = self._random_mask(rng, len(df), rate)
            df.loc[mask, column] = pd.NA
        return df


class ShifterVerb(BreakerVerb):
    supported_data_types = {"int", "float", "numeric", "string"}

    def _apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        data_type: str,
        rate: float,
        columns: list[str],
    ) -> pd.DataFrame:
        for column in columns:
            mask = self._random_mask(rng, len(df), rate)
            if data_type in {"int", "numeric"} and is_integer_dtype(df[column]):
                deltas = rng.integers(-5, 6, size=int(mask.sum()))
                df.loc[mask, column] = df.loc[mask, column] + deltas
                continue

            if data_type in {"float", "numeric"} and is_float_dtype(df[column]):
                deltas = rng.uniform(-5.0, 5.0, size=int(mask.sum()))
                df.loc[mask, column] = df.loc[mask, column] + deltas
                continue

            if data_type == "string":
                values = df.loc[mask, column]
                if len(values) > 1:
                    shuffled = values.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
                    df.loc[mask, column] = shuffled.to_numpy()
        return df


class DateJitterVerb(BreakerVerb):
    supported_data_types = {"date", "datetime"}

    def _apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        data_type: str,
        rate: float,
        columns: list[str],
    ) -> pd.DataFrame:
        for column in columns:
            mask = self._random_mask(rng, len(df), rate)
            count = int(mask.sum())
            if count == 0:
                continue

            if data_type == "datetime":
                offsets = pd.to_timedelta(rng.integers(-3600, 3601, size=count), unit="s")
                df.loc[mask, column] = pd.to_datetime(df.loc[mask, column]) + offsets
                continue

            offsets = pd.to_timedelta(rng.integers(-2, 3, size=count), unit="D")
            jittered = pd.to_datetime(df.loc[mask, column]) + offsets
            df.loc[mask, column] = jittered.dt.date
        return df


class ScramblerVerb(BreakerVerb):
    supported_data_types = {"string"}

    def _apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        data_type: str,
        rate: float,
        columns: list[str],
    ) -> pd.DataFrame:
        for column in columns:
            mask = self._random_mask(rng, len(df), rate)
            df.loc[mask, column] = df.loc[mask, column].map(lambda value: self._scramble_string_value(rng, value))
        return df


class CaserVerb(BreakerVerb):
    supported_data_types = {"string"}

    def _apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        data_type: str,
        rate: float,
        columns: list[str],
    ) -> pd.DataFrame:
        for column in columns:
            mask = self._random_mask(rng, len(df), rate)
            df.loc[mask, column] = df.loc[mask, column].map(lambda value: self._randomize_case_value(rng, value))
        return df


class ScalerVerb(BreakerVerb):
    supported_data_types = {"int", "float", "numeric"}

    def _apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        data_type: str,
        rate: float,
        columns: list[str],
    ) -> pd.DataFrame:
        for column in columns:
            mask = self._random_mask(rng, len(df), rate)
            count = int(mask.sum())
            if count == 0:
                continue

            factors = rng.uniform(0.0, 100.0, size=count)
            scaled_values = df.loc[mask, column] * factors
            if is_integer_dtype(df[column]):
                scaled_values = np.rint(scaled_values).astype("int64")
            df.loc[mask, column] = scaled_values
        return df
