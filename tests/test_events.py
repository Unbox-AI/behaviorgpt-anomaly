import warnings

from behaviorgpt import Order, Search, View

from anomaly.events import Event, to_sdk


def test_carting_is_a_view_and_an_order_holds_the_cart():
    events = [
        Event("search", query="headphones"),
        Event("view", item="a"),
        Event("add_to_cart", item="a"),
        Event("add_to_cart", item="b"),
        Event("add_to_cart", item="b"),
        Event("remove_from_cart", item="b"),
        Event("add_to_cart", item="c"),
        Event("remove_from_cart", item="c"),
        Event("order"),
        Event("add_to_cart", item="d"),
        Event("order"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sdk = to_sdk(events)
    kinds = [Search, View, View, View, View, View, Order, View, Order]
    assert [type(e) for e in sdk] == kinds
    assert [e.item for e in sdk[1:6]] == ["a", "a", "b", "b", "c"]
    first, second = sdk[6], sdk[8]
    assert [(i.item, i.quantity) for i in first.items] == [("a", 1), ("b", 1)]
    assert [(i.item, i.quantity) for i in second.items] == [("d", 1)]
