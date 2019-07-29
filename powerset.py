

from typing import *
from itertools import chain, combinations, count, accumulate
from functools import partial
from operator import contains
import random
from math import factorial

# Helper methods
def binomial(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


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
        return map(frozenset, chain.from_iterable(
            map(partial(combinations, self._items), range(0, len(self._items)+1))
        ))

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


    def random_samples(self, n: int=None, r: Union[int, Iterable[int]]=None):
        '''
        Creates an iterator that returns up to n random subsets of size r from this powerset (the same
        subset can be returned more than 1 time).

        :param n: Indicates the number of subsets to be returned.
            If not specified, the iterator will return an infinite number of subsets
        :param r: If specified, only returns subsets of size r if it is an integer number.
            It can also be an iterable of integer values. In that case, returns subsets with
            sizes equal to one of the items indicated

        e.g:
        iter(PowerSet(3).random_samples(r=1)) -> {0}, {1}, {0}, {2}, ...
        iter(PowerSet(3).random_samples(r=[1, 3])) -> {0}, {1}, {0, 1, 2}, {2}, {0}, ...

        list(PowerSet(3).random_samples(n=2, r=1)) -> [{1}, {0}]
        list(PowerSet().random_samples(n=2)) -> [{}, {}]
        '''
        if not (n is None or (isinstance(n, int) and n >= 0)):
            raise ValueError('Invalid value for argument n. Must be a number >= 0')

        m = len(self._items)
        if r is not None and not (isinstance(r, int) and r in range(0, m+1)) and\
            not (isinstance(r, Iterable) and all(map(partial(contains, range(0, m+1)), r))):
            raise ValueError('Invalid value for argument r')

        if r is None:
            r = range(0, m+1)
        elif isinstance(r, Iterable):
            r = frozenset(r)

        return self._random_samples(n, r)


    def _random_samples(self, n, r):
        m = len(self._items)
        n = count() if n is None else range(0, n)

        if isinstance(r, int):
            items = tuple(self._items)
            for x in n:
                yield frozenset(map(items.__getitem__, sorted(random.sample(range(0, m), r))))
        else:
            r = tuple(r)
            if len(r) == 0:
                return

            pool = tuple(map(partial(self._random_samples, None), r))
            sc = tuple(accumulate(map(partial(binomial, m), r)))

            for x in n:
                v = random.randrange(0, sc[-1])
                for k in range(0, len(r)):
                    if v < sc[k]:
                        yield next(pool[k])
                        break



    def random_sample(self, r=None):
        '''
        Its equivalent to next(self.random(n=1, r)).
        '''
        return next(self.random_samples(1, r))


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
        return True if isinstance(other, PowerSet) else super().isdisjoint(other)

    def __str__(self):
        return '{' + ', '.join(map(lambda s: str(set(s)), self)) + '}'

    def __repr__(self):
        return str(self)



if __name__ == '__main__':
    # Run this module as script to execute the unitary test

    from itertools import product, repeat, starmap
    from operator import eq
    import unittest
    from unittest import TestCase
    from random import sample

    # Random distribution tests (only run if numpy and scipy are avaliable)
    try:
        import numpy
        import scipy
    except:
        pass


    class TestPowerSet(TestCase):
        def test_iterator(self):
            # set(PowerSet()) -> { {} }
            self.assertEqual(set(PowerSet()), set([frozenset()]))

            # set(PowerSet(1)) -> { {}, {0} }
            self.assertEqual(set(PowerSet(1)), { frozenset(), frozenset([0]) })

            # set(PowerSet(2)) -> { {}, {0}, {1}, {0, 1} }
            self.assertEqual(set(PowerSet(2)),
                set(map(frozenset, [set(), {0}, {1}, {0, 1}])))

            # set(PowerSet(3)) -> { {}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2} }
            self.assertEqual(set(PowerSet(3)),
                set(map(frozenset, [set(), {0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}, {0, 1, 2}])))


        def test_constructor(self):
            # PowerSet(x) <=> PowerSet(range(0, x))  where x is a number > 0
            for k in range(1, 6):
                self.assertEqual(set(PowerSet(k)), set(PowerSet(range(0, k))))

            # PowerSet(X) <=> PowerSet(frozenset(X)) where X is a iterable
            for k in range(1, 6):
                self.assertEqual(set(PowerSet(list(range(k, 6)) * 2)), set(PowerSet(range(k, 6))))


        def test_length(self):
            # len(PowerSet()) == 1
            self.assertEqual(len(PowerSet()), 1)

            # len(PowerSet(X)) == len(list(PowerSet(X)))
            for k in range(1, 6):
                self.assertEqual(len(PowerSet()), len(list(PowerSet())))


        def test_contains(self):
            # for all item y in list(PowerSet(X)) if and only if y in PowerSet(x) == True
            for k in range(1, 6):
                s = PowerSet(k)
                for y in s:
                    self.assertIn(y, s)
                    if len(y) > 0:
                        self.assertNotIn(set(map(lambda x: x + k, y)), s)


        def test_hash(self):
            # hash(PowerSet(X)) = hash(PowerSet(Y)) <=> hash(tuple(PowerSet(X))) = hash(tuple(PowerSet(Y)))
            for i, j in product(range(1, 6), range(1, 6)):
                s, t = PowerSet(i), PowerSet(j)
                self.assertFalse((hash(s) == hash(t)) ^ (hash(tuple(s)) == hash(tuple(t))))


        def test_comparisons(self):
            # PowerSet(X) == PowerSet(Y) <=> hash(PowerSet(X)) == hash(PowerSet(Y))
            # PowerSet(X) < PowerSet(Y) <=> set(PowerSet(X)) < set(PowerSet(Y))
            # PowerSet(X) <= PowerSet(Y) <=> set(PowerSet(X)) <= set(PowerSet(Y))
            # PowerSet(X) > PowerSet(Y) <=> set(PowerSet(X)) > set(PowerSet(Y))
            # PowerSet(X) >= PowerSet(Y) <=> set(PowerSet(X)) >= set(PowerSet(Y))

            self.assertEqual(PowerSet(), PowerSet())
            self.assertFalse(PowerSet() < PowerSet())

            for i, j in product(range(1, 6), range(1, 6)):
                s, t = PowerSet(i), PowerSet(j)
                self.assertFalse((hash(s) == hash(t)) ^ (s == t))
                self.assertFalse((set(s) < set(t)) ^ (s < t))
                self.assertFalse((set(s) <= set(t)) ^ (s <= t))
                self.assertFalse((set(s) > set(t)) ^ (s > t))
                self.assertFalse((set(s) >= set(t)) ^ (s >= t))

        def test_disjoint(self):
            # PowerSet(X).isdisjoint(PowerSet(Y)) = True
            for i, j in product(range(1, 6), range(1, 6)):
                self.assertTrue(PowerSet(i).isdisjoint(PowerSet(j)))

        def test_random_samples(self):
            try:
                import numpy as np
                from scipy.stats import shapiro, norm

                # len(list(PowerSet(X).random_samples(n, r))) == n
                # len(s) == r for any s in PowerSet(X).random_samples(n, r)
                for i in range(1, 6):
                    s = PowerSet(i)
                    for n, r in product(range(0, 20), range(0, i+1)):
                        samples = list(s.random_samples(n, r))
                        self.assertEqual(len(samples), n)
                        self.assertTrue(all(starmap(eq, zip(map(len, samples), repeat(r)))))

                # len(list(PowerSet(X).random_samples(n, []))) == 0
                for k in range(1, 6):
                    self.assertEqual(len(list(PowerSet(k).random_samples(r=[]))), 0)

                # number of samples of size r in PowerSet(X).random_samples(n)
                # is binomial(n, r) / 2**len(PowerSet(X)) on average

                # if X = list(map(len, PowerSet(k).random_samples(n))) with n > 30
                # X ~ B(k/2, 0.5) ~ N(k / 2, k / 4)

                n = 2000
                for k in range(1, 6):
                    X = np.array(list(map(len, PowerSet(k).random_samples(n))))

                    # X is normally distributed
                    self.assertLessEqual(shapiro(X)[1], 0.05)

                    # X ~ N(k / 2, k / 4)
                    s = k / (2 * np.sqrt(n))
                    z = (np.mean(X) - k / 2) / s
                    p = 2 * (1 - norm.cdf(np.abs(z)))
                    self.assertGreater(p, 0.05)
            except ModuleNotFoundError:
                pass

    unittest.main()
