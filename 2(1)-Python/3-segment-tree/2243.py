from lib import SegmentTree
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