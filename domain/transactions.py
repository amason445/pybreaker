from pydantic import BaseModel, Field, model_validator
from dataclasses import dataclass

from uuid import UUID
from datetime import datetime, date
from decimal import Decimal

class RetailTransaction(BaseModel):
    """Unused transaction header model kept for future rollups."""
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
    """Unused transaction line model kept for future normalized outputs."""
    line_id: UUID
    transaction_id: UUID
    item_id: UUID
    quantity: int
    unit_price: Decimal
    discount_amount: Decimal = Decimal("0.00")

class RawRetailLine(BaseModel):
    """Validated line-level retail transaction record.

    This is the primary output model used by the synthetic generator and later
    converted into pandas records for breaker-based mutation.
    """

    transaction_id: UUID
    line_id: UUID
    store_id: UUID
    register_id: UUID
    item_id: UUID
    payment_type: str
    transaction_ts: datetime
    business_date: date
    quantity: int = Field(ge=1)
    unit_price_cents: int = Field(ge=0)
    discount_cents: int = Field(ge=0)
    # capped at 20%
    tax_rate_bps: int = Field(ge=0, le=2000)
    extended_cents: int = Field(ge=0)
    net_cents: int = Field(ge=0)
    tax_cents: int = Field(ge=0)
    line_total_cents: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_amounts(self) -> "RawRetailLine":
        """Enforce key accounting relationships across cents-based fields."""
        expected_extended = self.quantity * self.unit_price_cents
        if self.extended_cents != expected_extended:
            raise ValueError("extended_cents must equal quantity * unit_price_cents")

        if self.discount_cents > self.extended_cents:
            raise ValueError("discount_cents cannot exceed extended_cents")

        expected_net = self.extended_cents - self.discount_cents
        if self.net_cents != expected_net:
            raise ValueError("net_cents must equal extended_cents - discount_cents")

        expected_total = self.net_cents + self.tax_cents
        if self.line_total_cents != expected_total:
            raise ValueError("line_total_cents must equal net_cents + tax_cents")

        return self

@dataclass(frozen=True)
class Item:
    """Synthetic item catalog entry used when sampling transactions."""
    item_id: UUID
    price_cents: int
    taxable: bool
    weight: float
