"""Shopper events as plain JSON-friendly records, and their SDK form."""

from dataclasses import asdict, dataclass
from typing import Literal

from behaviorgpt import Order, Search, View

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


def to_sdk(events: list[Event]) -> list:
    """The history as the model reads it.

    The model sees carting an item as viewing it and ignores removals; an order
    holds what is in the cart when it is placed.
    """
    sdk: list = []
    cart: dict[str, int] = {}
    for e in events:
        match e.type:
            case "search":
                sdk.append(Search(e.query))
            case "view":
                sdk.append(View(e.item))
            case "add_to_cart":
                sdk.append(View(e.item))
                cart[e.item] = cart.get(e.item, 0) + 1
            case "remove_from_cart":
                if cart.get(e.item, 0) > 1:
                    cart[e.item] -= 1
                else:
                    cart.pop(e.item, None)
            case "order":
                sdk.append(Order(list(cart.items())))
                cart = {}
            case _:
                raise ValueError(f"unknown event type {e.type!r}")
    return sdk


def key(events: list[Event]) -> str:
    """A stable text form of a history, used as a cache key."""
    return "|".join(
        f"{e.type}:{e.query if e.type == 'search' else e.item or ''}" for e in events
    )
