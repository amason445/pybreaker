import pandas as pd

from breakers.breaker import Breaker
from generators.transaction_generator import TransactionGenerator

# Generate a clean synthetic transaction dataset, then create a broken copy
# using reusable breaker verbs that mutate selected column types.
generator = TransactionGenerator(seed=42, store_number=2)
breaker = Breaker(seed=42)

store_dict = generator.get_store_dict()
global_item_list = generator.get_item_list()
global_transactions = generator.generate_transactions()

records = [line.model_dump() for line in global_transactions]
df_txns = pd.DataFrame(records)
df_txns = df_txns.sort_values(["transaction_id", "line_id", "transaction_ts"])

def build_mutated_frame(df: pd.DataFrame, rate_multiplier: float) -> pd.DataFrame:
    """Apply the full breaker suite at a chosen intensity level."""
    mutated_df = breaker.mutate(df, verb="KeyStripper", data_type="uuid", rate=0.03 * rate_multiplier)
    mutated_df = breaker.mutate(mutated_df, verb="Nuller", data_type="string", rate=0.015 * rate_multiplier, columns=["payment_type"])
    mutated_df = breaker.mutate(mutated_df, verb="doubler", data_type="numeric", rate=0.02 * rate_multiplier)
    mutated_df = breaker.mutate(mutated_df, verb="Shifter", data_type="numeric", rate=0.015 * rate_multiplier)
    mutated_df = breaker.mutate(mutated_df, verb="DateJitter", data_type="datetime", rate=0.01 * rate_multiplier, columns=["transaction_ts"])
    mutated_df = breaker.mutate(mutated_df, verb="DateJitter", data_type="date", rate=0.01 * rate_multiplier, columns=["business_date"])
    mutated_df = breaker.mutate(mutated_df, verb="scrambler", data_type="string", rate=0.01 * rate_multiplier, columns=["payment_type"])
    mutated_df = breaker.mutate(mutated_df, verb="caser", data_type="string", rate=0.02 * rate_multiplier, columns=["payment_type"])
    mutated_df = breaker.mutate(mutated_df, verb="scaler", data_type="numeric", rate=0.01 * rate_multiplier)
    return mutated_df


df_txns_low = build_mutated_frame(df_txns, rate_multiplier=1.0)
df_txns_medium = build_mutated_frame(df_txns, rate_multiplier=2.0)
df_txns_high = build_mutated_frame(df_txns, rate_multiplier=4.0)

df_txns.to_csv("synthetic_transactions.csv", index=False)
df_txns_low.to_csv("synthetic_transactions_broken_low.csv", index=False)
df_txns_medium.to_csv("synthetic_transactions_broken_medium.csv", index=False)
df_txns_high.to_csv("synthetic_transactions_broken_high.csv", index=False)

print(store_dict)
print(global_item_list)
print(df_txns.head())
print(df_txns_low.head())
print(df_txns_medium.head())
print(df_txns_high.head())
