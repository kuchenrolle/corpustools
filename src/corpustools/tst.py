class Node():
    __slots__ = ("char", "count", "lo", "eq", "hi")

    def __init__(self, char):
        self.char = char
        self.count = 0

        self.lo = None
        self.eq = None
        self.hi = None


class TernarySearchTree():
    """Ternary search tree that stores counts for n-grams
    and their subsequences.
    """

    def __init__(self, splitchar=None):
        """Initializes TST.

        Parameters
        ----------
        splitchar : str
            Character that separates tokens in n-gram.
            Counts are stored for complete n-grams and
            each sub-sequence ending in this character
        """
        self._root = None
        self._splitchar = splitchar
        self._total = 0

    def insert(self, string):
        """Insert string into Tree.

        Parameters
        ----------
        string : str
            String to be inserted.
        """
        self._root = self._insert(string, self._root)
        self._total += 1

    def frequency(self, string):
        """Return frequency of string.

        Parameters
        ----------
        string : str


        Returns
        -------
        int
            Frequency

        Notes
        -----
        Does not return substring frequency.
        Can be obtained by summing over completions.
        """
        if not string:
            return self._total

        node = self._search(string, self._root)

        if not node:
            return 0

        return node.count

    def completions(self, prefix="", full=True, return_frequency=True):
        """Return all completions for a given prefix.

        Parameters
        ----------
        prefix : str
            String that all results returned begin with.
        full : bool
            Flag for whether to return results with the prefix appended.
        return_frequency : bool
            If true, results will include frequency of each completion.

        Returns
        -------
        Generator
            Yield str or (str, int)-tuples (return_frequency=True)
        """
        prefix_node = self._search(prefix, self._root)

        if not prefix_node:
            return

        if prefix:
            prefix_node = prefix_node.eq

        for completion, frequency in self._completions(prefix_node):
            if full:
                completion = prefix + completion

            if return_frequency:
                yield completion, frequency
            else:
                yield completion

    def _insert(self, string, node):
        """Insert string at a given node.
        """
        if not string:
            return node

        char, *rest = string

        if node is None:
            node = Node(char)

        if char == node.char:
            if not rest:
                node.count += 1
                return node
            else:
                if rest[0] == self.splitchar:
                    node.count += 1
                node.eq = self._insert(rest, node.eq)

        elif char < node.char:
            node.lo = self._insert(string, node.lo)

        else:
            node.hi = self._insert(string, node.hi)

        return node

    def _search(self, string, node):
        """Return node that string ends in.
        """
        if not string or not node:
            return node

        char, *rest = string

        if char == node.char:
            if not rest:
                return node
            return self._search(rest, node.eq)

        elif char < node.char:
            return self._search(string, node.lo)

        else:
            return self._search(string, node.hi)

    def _completions(self, node):
        """Generator yielding completions starting from node.
        """
        if node is None:
            return

        if node.count:
            yield node.char, node.count

        if node.eq:
            for completion, frequency in self._completions(node.eq):
                yield node.char + completion, frequency

        if node.lo:
            for completion in self._completions(node.lo):
                yield completion

        if node.hi:
            for completion in self._completions(node.hi):
                yield completion

    def __contains__(self, string):
        """Adds "string in TST" syntactic sugar.
        """
        node = self._search(string, self._root)
        if node:
            return node.count

        return False

    @property
    def splitchar(self):
        return self._splitchar
