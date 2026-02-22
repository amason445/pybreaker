import pandas as pd

from generators.transaction_generator import TransactionGenerator

generator = TransactionGenerator(seed=42, store_number=2)

store_dict = generator.get_store_dict()
global_item_list = generator.get_item_list()
global_transactions = generator.generate_transactions()

records = [line.model_dump() for line in global_transactions]
df = pd.DataFrame(records)
df = df.sort_values(["transaction_id", "line_id", "transaction_ts"])
df.to_csv("synthetic_transactions.csv", index=False)

print(store_dict)
print(global_item_list)
print(df.head())