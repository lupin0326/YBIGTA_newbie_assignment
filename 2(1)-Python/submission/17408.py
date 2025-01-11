from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T]):
    def __init__(self, size: int, function: Callable[[T, T], T], default: T):
        """
        Initialize the Segment Tree.

        Args:
        - size (int): The maximum size of the input data.
        - function (Callable[[T, T], T]): The associative function to apply (e.g., sum, min, max).
        - default (T): The default value for unused nodes.
        """
        self.size = size
        self.function = function
        self.default = default
        self.tree = [default] * (4 * size)

    def update(self, index: int, value: T):
        """
        Update the value at a specific index and reflect the change in the tree.

        Args:
        - index (int): The index to update.
        - value (T): The new value to set.
        """
        self._update(0, 0, self.size - 1, index, value)

    def _update(self, node: int, start: int, end: int, index: int, value: T):
        if start == end:
            # Leaf node
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2

            if index <= mid:
                # Update in the left subtree
                self._update(left_child, start, mid, index, value)
            else:
                # Update in the right subtree
                self._update(right_child, mid + 1, end, index, value)

            # Update current node
            self.tree[node] = self.function(self.tree[left_child], self.tree[right_child])

    def query(self, l: int, r: int) -> T:
        """
        Query the result of the function over the range [l, r].

        Args:
        - l (int): The left bound of the range (inclusive).
        - r (int): The right bound of the range (inclusive).

        Returns:
        - T: The result of the function over the range.
        """
        return self._query(0, 0, self.size - 1, l, r)

    def _query(self, node: int, start: int, end: int, l: int, r: int) -> T:
        if r < start or l > end:
            # Range is completely outside
            return self.default
        if l <= start and end <= r:
            # Range is completely inside
            return self.tree[node]

        # Partial overlap
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        left_result = self._query(left_child, start, mid, l, r)
        right_result = self._query(right_child, mid + 1, end, l, r)

        return self.function(left_result, right_result)



import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


class Pair(tuple[int, int]):
    """
    힌트: 2243, 3653에서 int에 대한 세그먼트 트리를 만들었다면 여기서는 Pair에 대한 세그먼트 트리를 만들 수 있을지도...?
    """
    def __new__(cls, a: int, b: int) -> 'Pair':
        return super().__new__(cls, (a, b))

    @staticmethod
    def default() -> 'Pair':
        """
        기본값
        이게 왜 필요할까...?
        """
        return Pair(0, 0)

    @staticmethod
    def f_conv(w: int) -> 'Pair':
        """
        원본 수열의 값을 대응되는 Pair 값으로 변환하는 연산
        이게 왜 필요할까...?
        """
        return Pair(w, 0)

    @staticmethod
    def f_merge(a: Pair, b: Pair) -> 'Pair':
        """
        두 Pair를 하나의 Pair로 합치는 연산
        이게 왜 필요할까...?
        """
        return Pair(*sorted([*a, *b], reverse=True)[:2])

    def sum(self) -> int:
        return self[0] + self[1]


def main() -> None:
    # 구현하세요!
    pass


if __name__ == "__main__":
    main()