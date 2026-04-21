from __future__ import annotations

import numpy as np
import pandas as pd

from breakers.verbs import (
    BreakerVerb,
    CaserVerb,
    DateJitterVerb,
    DoublerVerb,
    KeyStripperVerb,
    NullerVerb,
    ScalerVerb,
    ScramblerVerb,
    ShifterVerb,
)


class Breaker:
    """Factory-style dispatcher for column-oriented DataFrame mutation verbs.

    This class follows a Java-style factory pattern: it owns a registry of verb
    implementations and creates/selects the appropriate verb object based on the
    requested verb name.

    The public API is still organized around `verb + data_type`, for example:

    - `mutate(df, verb="KeyStripper", data_type="uuid")`
    - `mutate(df, verb="doubler", data_type="numeric")`

    If `columns` is omitted, the breaker scans the frame and targets every
    matching column for the requested data type. If `columns` is provided, the
    breaker validates that each named column exists and matches the requested
    type before mutating it.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self._verb_registry: dict[str, type[BreakerVerb]] = {
            "keystripper": KeyStripperVerb,
            "doubler": DoublerVerb,
            "nuller": NullerVerb,
            "shifter": ShifterVerb,
            "datejitter": DateJitterVerb,
            "scrambler": ScramblerVerb,
            "caser": CaserVerb,
            "scaler": ScalerVerb,
        }

    def mutate(
        self,
        df: pd.DataFrame,
        verb: str,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Create a verb via the factory registry and apply it to the DataFrame."""
        normalized_verb = verb.strip().lower()
        normalized_type = data_type.strip().lower()
        verb_instance = self.create_verb(normalized_verb)
        return verb_instance.apply(
            df,
            rng=self.rng,
            data_type=normalized_type,
            rate=rate,
            columns=columns,
            inplace=inplace,
        )

    def create_verb(self, verb: str) -> BreakerVerb:
        """Factory method that creates a concrete verb implementation by name."""
        normalized_verb = verb.strip().lower()
        verb_class = self._verb_registry.get(normalized_verb)
        if verb_class is None:
            raise ValueError(f"Unsupported verb: {verb}")
        return verb_class()

    def KeyStripper(
        self,
        df: pd.DataFrame,
        data_type: str = "uuid",
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Compatibility wrapper for the KeyStripper verb."""
        return self.create_verb("keystripper").apply(df, self.rng, data_type, rate, columns, inplace)

    def Doubler(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Compatibility wrapper for the doubler verb."""
        return self.create_verb("doubler").apply(df, self.rng, data_type, rate, columns, inplace)

    def Nuller(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Compatibility wrapper for the Nuller verb."""
        return self.create_verb("nuller").apply(df, self.rng, data_type, rate, columns, inplace)

    def Shifter(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Compatibility wrapper for the Shifter verb."""
        return self.create_verb("shifter").apply(df, self.rng, data_type, rate, columns, inplace)

    def DateJitter(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Compatibility wrapper for the DateJitter verb."""
        return self.create_verb("datejitter").apply(df, self.rng, data_type, rate, columns, inplace)

    def Scrambler(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Compatibility wrapper for the scrambler verb."""
        return self.create_verb("scrambler").apply(df, self.rng, data_type, rate, columns, inplace)

    def Caser(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Compatibility wrapper for the caser verb."""
        return self.create_verb("caser").apply(df, self.rng, data_type, rate, columns, inplace)

    def Scaler(
        self,
        df: pd.DataFrame,
        data_type: str,
        rate: float = 0.1,
        columns: list[str] | None = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Compatibility wrapper for the scaler verb."""
        return self.create_verb("scaler").apply(df, self.rng, data_type, rate, columns, inplace)
