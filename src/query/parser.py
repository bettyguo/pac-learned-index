"""Query parsing utilities for benchmark workloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class QueryType(Enum):
    """Supported query types for benchmark evaluation."""
    POINT = "point"
    RANGE = "range"


@dataclass
class Query:
    """A benchmark query."""
    query_type: QueryType
    key: float
    range_end: float | None = None

    @classmethod
    def point_query(cls, key: float) -> "Query":
        return cls(query_type=QueryType.POINT, key=key)

    @classmethod
    def range_query(cls, key_start: float, key_end: float) -> "Query":
        return cls(query_type=QueryType.RANGE, key=key_start, range_end=key_end)


def parse_workload(queries_raw: list[dict[str, Any]]) -> list[Query]:
    """Parse a list of raw query dictionaries into Query objects."""
    parsed = []
    for q in queries_raw:
        qtype = QueryType(q.get("type", "point"))
        if qtype == QueryType.RANGE:
            parsed.append(Query.range_query(q["key"], q["range_end"]))
        else:
            parsed.append(Query.point_query(q["key"]))
    return parsed
