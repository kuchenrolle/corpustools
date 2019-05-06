# cython: language_level=3
cdef class Node():
    cdef:
        public Py_UNICODE character
        public unsigned int count
        public Node lo, eq, hi

    def __init__(self, Py_UNICODE character):
        self.character = character


cdef class TernarySearchTree():
    """Ternary search tree that stores counts for n-grams
    and their subsequences.
    """
    cdef:
        Node root
        Py_UNICODE _splitchar
        unsigned int total

    def __init__(self, splitchar=None):
        """Initializes TST.
        Parameters
        ----------
        splitchar : str
            Character that separates tokens in n-gram.
            Counts are stored for complete n-grams and
            each sub-sequence ending in this character
        """
        self._splitchar = splitchar

    cpdef void insert(self, str string):
        """Insert string into Tree.
        Parameters
        ----------
        string : str
            String to be inserted.
        """
        self.root = self._insert(string, self.root)
        self.total += 1

    cpdef int frequency(self, str string):
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
        cdef Node node

        if not string:
            return self.total

        node = self._search(string, self.root)

        if not node:
            return 0

        return node.count

    def completions(self, str prefix="", bint full=True,
                    bint return_frequency=True):
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
        cdef:
            Node prefix_node
            str completion
            int frequency

        prefix_node = self._search(prefix, self.root)

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

    cdef Node _insert(self, str string, Node node):
        """Insert string at a given node.
        """
        cdef:
            Py_UNICODE character
            str rest

        if not string:
            return node

        # character, *rest = string
        character, rest = string[0], string[1:]

        if node is None:
            node = Node(character)

        if character == node.character:
            if not rest:
                node.count += 1
                return node

            if rest[0] == self.splitchar:
                node.count += 1
            node.eq = self._insert(rest, node.eq)

        elif character < node.character:
            node.lo = self._insert(string, node.lo)

        else:
            node.hi = self._insert(string, node.hi)

        return node

    cdef Node _search(self, str string, Node node):
        """Return node that string ends in.
        """
        cdef:
            Py_UNICODE character
            str rest

        if not string or not node:
            return node

        # character, *rest = string
        character, rest = string[0], string[1:]

        if character < node.character:
            return self._search(string, node.lo)

        elif character > node.character:
            return self._search(string, node.hi)

        if not rest:
            return node

        return self._search(rest, node.eq)

    def _completions(self, Node node):
        """Generator yielding completions starting from node.
        """
        cdef str completion
        cdef int frequency

        if node is None:
            return

        if node.count:
            yield node.character, node.count

        if node.eq:
            for completion, frequency in self._completions(node.eq):
                yield node.character + completion, frequency

        if node.lo:
            for completion, frequency in self._completions(node.lo):
                yield completion, frequency

        if node.hi:
            for completion, frequency in self._completions(node.hi):
                yield completion, frequency

    def __contains__(self, str string):
        """Adds 'string in TST' syntactic sugar.
        """
        cdef Node node

        node = self._search(string, self.root)
        if node:
            return node.count

        return False

    def __iter__(self):
        """Adds 'for string in TST' syntactic sugar.
        """
        return self.completions()

    @property
    def splitchar(self):
        return self._splitchar

