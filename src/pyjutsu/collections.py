import itertools
from typing import Union, List


class _Circular:
    def __init__(self, lst: list):
        self.lst = lst

    def __getitem__(self, slice_or_idx: Union[int, slice]):
        """
        Some implementation notes.
        We represent circular collection as infinite concatenation of original
        sequence, e.g. `[1,2] => [1,2,1,2,1,2...]`
        Thus when we apply slice we do it as it would be applied to that infinite sequence.

        Let's imagine that each occurance of original sequence as a slot.
        E.g. result of concatenation `[1,2] => [1,2,1,2]` has two slots.

        Now we just should calculate slot of `start` and `stop` index.
        * If it is same slot, then we return `self.lst[start mod n : stop mod n : step]`
        * If slots differ, then we return `head + body + tail` sliced with `step`, where
           * `head = self.lst[start mod n:]`
           * `tail = self.lst[:stop mod n]`
           * `body = self.lst * (stop_slot - start_slot - 1)`
        """

        if isinstance(slice_or_idx, int):
            return self.lst[slice_or_idx % len(self.lst)]



        n = len(self.lst)
        start = slice_or_idx.start or 0
        stop = slice_or_idx.stop or n
        step = slice_or_idx.step

        assert stop >= start, "So far, we expect only straight forward indexing"

        if n == 1:
            repeat = stop - start
            if step:
                repeat = (repeat-1) // step + 1
            return itertools.repeat(self.lst[0], repeat)

        head_slot = start // n
        tail_slot = stop // n

        start_mod = start % n
        stop_mod = stop % n

        if head_slot == tail_slot:
            return self.lst[start_mod:stop_mod:step]

        head = self.lst[start_mod:]
        tail = self.lst[:stop_mod]

        num_inners = tail_slot - head_slot - 1
        body = itertools.chain(*[self.lst]*num_inners)

        return itertools.islice(itertools.chain(head, body, tail), None, None, step)

    def __setitem__(self, idx, value):
        assert isinstance(idx, int)
        self.lst[idx % len(self.lst)] = value


def circular(lst: List):
    """
    Implements a circular collection based on input list collection.
    It implements slice operators (e.g. ll[2:10:2]) for read-only mode
    Indexing operator (e.g. ll[2]) allows both read and write operations
    E.g. circular([1,2,3])[3] is 1, and circular([1,2,3])[4:5] is [2,3]
    :param lst: list which is used as a source
    :return: circular collection.
    """
    return _Circular(lst)
