"""Small serialization helpers for cloud repositories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class DumpableModel(Protocol):
    def model_dump(self, *, mode: str) -> dict[str, JsonValue]: ...


def model_to_json_dict(model: DumpableModel) -> dict[str, JsonValue]:
    return model.model_dump(mode="json")


def mapping_to_json_dict(value: Mapping[str, object]) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], dict(value))
