from pydantic import BaseModel, Field
from dataclasses import dataclass

from uuid import UUID
from datetime import datetime
from decimal import Decimal

class RetailTransaction(BaseModel):
    transaction_id: UUID
    store_id: UUID
    register_id: UUID
    customer_id: UUID | None
    transaction_datetime: datetime
    subtotal: Decimal
    tax_amount: Decimal
    total_amount: Decimal
    payment_method: str

class RetailTransactionLineItem(BaseModel):
    line_id: UUID
    transaction_id: UUID
    item_id: UUID
    quantity: int
    unit_price: Decimal
    discount_amount: Decimal = Decimal("0.00")

class RawRetailLine(BaseModel):
    transaction_id: UUID
    line_id: UUID
    store_id: UUID
    register_id: UUID
    item_id: UUID
    transaction_ts: datetime
    quantity: int = Field(ge=1)
    unit_price_cents: int = Field(ge=0)
    discount_cents: int = Field(ge=0)
    # capped at 20%
    tax_rate_bps: int = Field(ge=0, le=2000)
    extended_cents: int
    net_cents: int
    tax_cents: int
    line_total_cents: int

@dataclass(frozen=True)
class Item:
    item_id: UUID
    price_cents: int
    taxable: bool
    weight: float