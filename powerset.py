

from typing import *
from itertools import chain, combinations, filterfalse, starmap
from functools import partial
from operator import contains, le


T_co = TypeVar('T')
class PowerSet(AbstractSet[T_co], Hashable):
    '''
    Represents a set which contains all the subsets of a given set.
    '''

    def __init__(self, x: Union[Iterable[T_co], int]=None):
        '''
        :param x: Its a iterable with the elements to build the power set from (all items must be
        hashable and duplicates will be removed)
        It can also be an integer number greater than 0 or None (which is the default
        value).
        If x is a number, PowerSet(x) is equivalent to PowerSet(range(0, x))
        If x is None, PowerSet(x) will only include the empty set {}

        e.g:
        list(PowerSet([a, b, c])) -> [{}, {a,}, {b,}, {c,}, {a, b}, {a, c}, {b, c}, {a, b, c}]
        list(PowerSet(2)) -> [(), (0,), (1,), (0, 1)]
        list(PowerSet()) -> [()]

        Note that PowerSet([1, 2]) and PowerSet([2, 1]) are equivalent:
        For any iterable X, PowerSet(X) is equivalent to PowerSet(set(X))
        '''
        if x is not None and not isinstance(x, Iterable) and not (isinstance(x, int) and x > 0):
            raise TypeError('Argument must be an iterable, a number greater than zero or None')

        if isinstance(x, Iterable):
            x = frozenset(x)
        else:
            x = frozenset(range(0, x)) if isinstance(x, int) else frozenset()

        self._items = x


    def __contains__(self, s: Iterable[T_co]):
        '''
        Checks if the given item s is inside of this power set.
        '''
        if not isinstance(s, Iterable):
            return False
        try:
            s = frozenset(s)
        except:
            return False

        return s <= self._items


    def __iter__(self) -> Iterator[AbstractSet[T_co]]:
        '''
        Iterates over the power set, returning all its items as frozenset objects.
        If the power set was created like PowerSet([x1, x2, ..., xn]) and assuming
        that x1 != x2 != ... != xn:

        The first item will be ()
        After that, subsets of size 1 will be returned:
        {x1,}, {x2,}, ..., {xn,}

        Then, combinations of size 2 of the items x0, x1, ..., xn with no repetitions:
        {x1, x2}, {x1, x3}, ..., {xn-1, xn}
        The next following subsets are combinations of size 3, size 4, ..., size n-1, also with no repetitions.

        Finally, the last item will be {x1, x2, ..., xn}
        '''
        return map(frozenset, chain.from_iterable(combinations(self._items, r) for r in range(len(self._items)+1)))


    def __len__(self):
        '''
        Returns the cardinality of this power set.
        The power set of a set with n elements will have 2**n cardinality
        '''
        return 2**len(self._items)


    def __hash__(self):
        '''
        Returns the hash for this power set.
        '''
        return hash((self.__class__, self._items))


    def __eq__(self, other):
        return self._items == other._items if isinstance(other, PowerSet) else super().__eq__(other)

    def __lt__(self, other):
        return self._items < other._items if isinstance(other, PowerSet) else super().__lt__(other)

    def __le__(self, other):
        return self._items <= other._items if isinstance(other, PowerSet) else super().__le__(other)

    def __gt__(self, other):
        return self._items > other._items if isinstance(other, PowerSet) else super().__gt__(other)

    def __ge__(self, other):
        return self._items >= other._items if isinstance(other, PowerSet) else super().__ge__(other)

    def isdisjoint(self, other):
        return self._items.isdisjoint(other._items) if isinstance(other, PowerSet) else super().isdisjoint(other)

    def __str__(self):
        return '{' + ', '.join(map(lambda s: str(set(s)), self)) + '}'

    def __repr__(self):
        return str(self)
