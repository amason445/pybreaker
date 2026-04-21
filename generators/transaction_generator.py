import random
import numpy as np
from typing import List
from uuid import UUID, uuid4
from datetime import datetime, timedelta

from domain.transactions import RawRetailLine, Item

class TransactionGenerator:
    """Generate seeded synthetic retail transaction line items.

    The generator creates stores, registers, and a weighted item catalog, then
    samples line-level transactions with quantity, pricing, discount, and tax
    calculations. Output rows are validated by the `RawRetailLine` Pydantic
    model.
    """

    def __init__(self, seed: int | None = None, store_number: int = 2, registers_per_store: int = 4, transactions_per_register: int = 100, items_total: int = 25):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.number_of_stores = store_number
        self.number_of_items = items_total
        self.number_of_transactions = transactions_per_register
        self.registers_per_store = registers_per_store
        self.store_structure = self._generate_store_dict()
        self.global_item_list = self._generate_item_list()
        self.store_ids = list(self.store_structure.keys())
        self.item_ids = [it.item_id for it in self.global_item_list]
        self.item_weights = [it.weight for it in self.global_item_list]
        self.item_by_id = {it.item_id: it for it in self.global_item_list}

    def _generate_store_dict(self) -> dict[UUID, list[UUID]]:
        """Create store IDs and the register IDs that belong to each store."""
        store_dict: dict[UUID, list[UUID]] = {}
        for _ in range(self.number_of_stores):
            store_id = uuid4()
            registers = [uuid4() for _ in range(self.registers_per_store)]
            store_dict[store_id] = registers
        return store_dict
    
    def _draw_price_cents(self, mu=1050, sigma=350, min_price=50):
        """Sample an item price in cents from a truncated normal-like distribution."""
        while True:
            x = self.rng.normalvariate(mu, sigma)
            if x >= min_price:
                return int(round(x))
            
    def _draw_weight(self):
        """Sample an item popularity weight for weighted item selection."""
        return self.rng.paretovariate(1.5)
    
    def _generate_item_list(self):
        """Build the global item catalog used across all generated transactions."""
        item_list: List[Item] = []
        for _ in range(self.number_of_items):
            item = Item(
                item_id = uuid4(),
                price_cents= self._draw_price_cents(),
                taxable = self.rng.choice([True, False]),
                weight = self._draw_weight()
            )
            item_list.append(item)
        return item_list
    
    def _sample_item(self) -> Item:
        """Choose an item using the configured item weights."""
        item_id = self.rng.choices(self.item_ids, weights=self.item_weights, k=1)[0]
        return self.item_by_id[item_id]
    
    def _sample_quantity(self) -> int:
        """Sample a line quantity from a simple mixture of count distributions."""
        r = self.rng.random()
        if r < 0.7:
            return int(self.np_rng.poisson(1.5) + 1)
        elif r < 0.95:
            return int(self.np_rng.poisson(4) + 1)
        else:
            return int(self.np_rng.negative_binomial(5, 0.3) + 5)

    def _sample_discount_cents(self, extended_cents: int) -> int:
        """Apply no discount most of the time and otherwise sample a markdown."""
        if extended_cents <= 0:
            return 0

        # Most lines have no discount; discounted lines skew toward smaller markdowns.
        if self.rng.random() < 0.8:
            return 0

        discount_rate_bps = self.rng.randint(250, 3000)
        discount_cents = round(extended_cents * discount_rate_bps / 10_000)
        return min(discount_cents, extended_cents)

    def _sample_tax_rate_bps(self, taxable: bool) -> int:
        """Return a sampled tax rate in basis points for taxable items only."""
        if not taxable:
            return 0

        # Roughly 5% to 10% for taxable items, represented in basis points.
        return self.rng.randint(500, 1000)

    def _calculate_tax_cents(self, net_cents: int, tax_rate_bps: int) -> int:
        """Convert a net amount and tax rate into integer tax cents."""
        return round(net_cents * tax_rate_bps / 10_000)

    def _sample_payment_type(self) -> str:
        """Sample a payment type with a simple retail-style distribution."""
        payment_types = ["credit", "debit", "cash", "mobile_wallet", "gift_card"]
        weights = [0.4, 0.28, 0.15, 0.12, 0.05]
        return self.rng.choices(payment_types, weights=weights, k=1)[0]
        
    def get_store_dict(self):
        """Expose the generated store/register structure."""
        return self.store_structure
    
    def get_item_list(self):
        """Expose the generated global item catalog."""
        return self.global_item_list
    
    def generate_transactions(self, min_lines: int = 1, max_lines: int = 12) -> List[RawRetailLine]:
        """Generate synthetic retail line items and return them as validated models."""
        lines: List[RawRetailLine] = []
        total_transactions = self.number_of_stores * self.registers_per_store * self.number_of_transactions
        start_ts = datetime.now()
        end_ts = start_ts + timedelta(days=2)
        seconds_range = int((end_ts - start_ts).total_seconds())
        if seconds_range <= 0:
            raise ValueError("end_ts must be after start_ts")

        for _ in range(total_transactions):
            transaction_id = uuid4()
            store_id = self.rng.choice(self.store_ids)
            register_id = self.rng.choice(self.store_structure[store_id])
            payment_type = self._sample_payment_type()

            k = self.rng.randint(min_lines, max_lines)

            for _line in range(k):
                item = self._sample_item()
                qty = int(self._sample_quantity())

                offset_s = self.rng.randint(0, seconds_range)
                ts = start_ts + timedelta(seconds=offset_s)

                unit_price_cents = int(item.price_cents)
                extended_cents = qty * unit_price_cents

                discount_cents = self._sample_discount_cents(extended_cents)
                net_cents = extended_cents - discount_cents
                tax_rate_bps = self._sample_tax_rate_bps(item.taxable)
                tax_cents = self._calculate_tax_cents(net_cents, tax_rate_bps)
                line_total_cents = net_cents + tax_cents
                
                lines.append(
                    RawRetailLine(
                        transaction_id=transaction_id,
                        line_id=uuid4(),
                        store_id=store_id,
                        register_id=register_id,
                        item_id=item.item_id,
                        payment_type=payment_type,
                        transaction_ts=ts,
                        business_date=ts.date(),
                        quantity=qty,
                        unit_price_cents=unit_price_cents,
                        discount_cents=discount_cents,
                        tax_rate_bps=tax_rate_bps,
                        extended_cents=extended_cents,
                        net_cents=net_cents,
                        tax_cents=tax_cents,
                        line_total_cents=line_total_cents
                    )
                )
        return lines

    
    
