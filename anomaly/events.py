"""Shopper events as plain JSON-friendly records, and their SDK form."""

from dataclasses import asdict, dataclass
from typing import Literal

from behaviorgpt import AddToCart, Order, RemoveFromCart, Search, View

EventType = Literal["search", "view", "add_to_cart", "remove_from_cart", "order"]
ITEM_EVENTS = ("view", "add_to_cart", "remove_from_cart")


@dataclass(frozen=True)
class Event:
    type: EventType
    item: str | None = None
    query: str | None = None

    def __post_init__(self) -> None:
        if self.type in ITEM_EVENTS and not self.item:
            raise ValueError(f"{self.type} event needs an item id")
        if self.type == "search" and not self.query:
            raise ValueError("search event needs a query")

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        return cls(type=data["type"], item=data.get("item"), query=data.get("query"))

    def to_sdk(self):
        match self.type:
            case "search":
                return Search(self.query)
            case "view":
                return View(self.item)
            case "add_to_cart":
                return AddToCart(self.item)
            case "remove_from_cart":
                return RemoveFromCart(self.item)
            case "order":
                # the SDK fills the order with the cart built from earlier events
                return Order()
        raise ValueError(f"unknown event type {self.type!r}")


def key(events: list[Event]) -> str:
    """A stable text form of a history, used as a cache key."""
    return "|".join(
        f"{e.type}:{e.query if e.type == 'search' else e.item or ''}" for e in events
    )
