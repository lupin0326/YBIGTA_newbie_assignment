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

def main() -> None:
    input = sys.stdin.read
    data = input().splitlines()

    n = int(data[0])
    commands = data[1:]

    max_flavor = 1_000_000
    candy_tree = SegmentTree(max_flavor, function=lambda x, y: x + y, default=0)

    def add_candy(flavor: int, count: int):
        current_count = candy_tree.query(flavor - 1, flavor - 1)
        candy_tree.update(flavor - 1, current_count + count)

    def get_candy(rank: int) -> int:
        start, end = 0, max_flavor - 1
        node = 0

        while start != end:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2

            if candy_tree.tree[left_child] >= rank:
                node = left_child
                end = mid
            else:
                rank -= candy_tree.tree[left_child]
                node = right_child
                start = mid + 1

        flavor = start + 1
        add_candy(flavor, -1)  # Remove the candy
        return flavor

    results = []
    for command in commands:
        parts = list(map(int, command.split()))
        if parts[0] == 1:
            # Get the candy
            rank = parts[1]
            results.append(get_candy(rank))
        elif parts[0] == 2:
            # Add/remove candies
            flavor, count = parts[1], parts[2]
            add_candy(flavor, count)

    sys.stdout.write("\n".join(map(str, results)) + "\n")

if __name__ == "__main__":
    main()