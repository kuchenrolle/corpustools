from collections import deque
from .tst import TernarySearchTree
from .corpustools import extract_fields, ContainsEverything


class LanguageModel():
    """N-gram (Markov) model that uses a ternary search tree.
    Tracks frequencies and calculates probabilities.

    Attributes
    ----------
    n : int
        Size of n-grams to be tracked.
    vocabulary : set
        If provided, n-grams containing words not in vocabulary are skipped.
        Can be other container than set, if it has add method.
    targets : container
        If provided, n-grams not ending in target are counted as
        ending in "OOV" (OutOfVocabulary) instead, so probabilities
        can still be calculated.
    must_contain : container
        If provided, only n-grams containing at least one word in
        must_contain are counted
    boundary : str
        N-grams crossing boundary will not be counted,
        e.g. sentence </s> or document </doc> meta tags
    splitchar : str
        String that separates tokens in n-grams
    """

    def __init__(self, n, boundary="</s>", splitchar="#",
                 vocabulary=None, targets=None, must_contain=None):
        """
        Parameters
        ----------
        n : int
            Size of n-grams to be tracked.
        boundary : str
            N-grams crossing boundary will not be counted,
            e.g. sentence </s> or document </doc> meta tags
        splitchar : str
            String that separates tokens in n-grams
        vocabulary : set
            If provided, n-grams with words not in vocabulary are skipped.
            Can be other container than set, if it has add method.
        targets : container
            If provided, n-grams not ending in target are counted as
            ending in "OOV" (OutOfVocabulary) instead, so probabilities
            can still be calculated.
        must_contain : container
            If provided, only n-grams containing at least one word in
            must_contain are counted

        Notes
        -----
        If must_contain is provided, probabilities will be inaccurate. Only
        use for counting target n-gram frequencies.
        """
        if not targets:
            targets = ContainsEverything()

        if not vocabulary:
            vocabulary = ContainsEverything()

        self._n = n
        self._counts = TernarySearchTree(splitchar)
        self._vocabulary = vocabulary
        self._targets = targets
        self._boundary = boundary
        self._splitchar = splitchar
        self._must_contain = must_contain

    def train(self, sequence):
        """Train model on all n-grams in sequence.

        Parameters
        ----------
        sequence : iterable of str
            Sequence of tokens to train on.

        Notes
        -----
        A sequence [A, B, C, D, E] with n==3 will result in these
        n-grams:
          [A, B, C]
          [B, C, D]
          [C, D, E]
          [D, E]
          [E]
        """
        n_gram = deque(maxlen=self.n)
        for element in sequence:
            if element == self.boundary:
                # train on smaller n-grams at end of sentence
                # but exclude full n_gram if it was already trained
                # on in last iteration
                not_trained = len(n_gram) < self.n
                for length in range(1, len(n_gram) + not_trained):
                    self._train(list(n_gram)[-length:])
                n_gram.clear()
                continue

            n_gram.append(element)

            if len(n_gram) == self.n:
                if element not in self.targets:
                    self._train(list(n_gram)[:-1])
                    continue

                self._train(n_gram)

        # train on last n-grams in sequence
        # ignore full n-gram if it has already been trained on
        if len(n_gram) == self.n:
            n_gram = list(n_gram)[1:]

        for length in range(1, len(n_gram) + 1):
            self._train(list(n_gram)[-length:])

    def insert_sequence(self, counts,
                        is_string=True, subsequences=False):
        """Increase counts of sequence of ngrams by their frequencies.

        Parameters
        ----------
        counts : sequence of (str, int) tuples
            Tuples of ngrams and their frequencies
        is_string : bool
            If True, ngrams are assumed to be strings.
            Otherwise they are assumed to be tuples
            of strings, which will be joined by self.splitchar.
        subsequences : bool
            If True, counts for subsequences of n-gram will
            also be increased by frequency. A subsequence
            is everything that ends in self.splitchar,
            e.g. for "my#shiny#trigram", subsequences are
            "my#shiny" and "my"
        """
        for ngram, frequency in counts:
            self.insert(ngram, frequency, is_string, subsequences)

    def insert(self, ngram, frequency,
               is_string=True, subsequences=False):
        """Increases count of n-gram by frequency.

        Parameters
        ----------
        ngram : str or sequence of str
            n-gram as string or sequence of strings (words)
        frequency : int
            Frequency of n-gram
        is_string : bool
            If True, n-gram must be a string, with
            self.splitchar (default '#') separating words.
        subsequences : bool
            If True, counts for subsequences of n-gram will
            also be increased by frequency. A subsequence
            is everything that ends in self.splitchar,
            e.g. for "my#shiny#trigram", subsequences are
            "my#shiny" and "my"
        """
        if not is_string:
            ngram = self.splitchar.join(ngram)

        self._counts.insert(ngram, frequency,
                            subsequences)

    def probability(self, sequence, predict_all=False):
        """Returns probability of the sequence.

        Parameters
        ----------
        sequence : iterable of str
            Sequence of tokens to get the probability for
        predict_all : bool
            Return probability for each word in the sequence (True)
            or only for the last word (False).

        Returns
        -------
        float or list of float
            Probability of last element or probabilities of all elements
        """
        n_gram = deque(maxlen=self.n)

        if predict_all:
            probabilities = []
            for element in sequence:
                n_gram.append(element)
                probability = self._probability(n_gram)
                probabilities.append(probability)
            return probabilities

        else:
            try:
                n_gram.extend(sequence[-self.n:])

            # if sequence is generator (cannot slice - TypeError),
            # run through it and return probability for final element
            except TypeError:
                for element in sequence:
                    n_gram.append(element)

            probability = self._probability(n_gram)
            return probability

    def all_target_probabilities(self, return_n_gram=False, sizes=None):
        """Generator yielding probabilities and frequencies
        of all encountered targets.

        Parameters
        ----------
        return_n_gram : bool
            Return full n-gram rather than just target with results
        sizes: list of int
            Sizes of n-grams to be returned, defaults to target size

        Returns
        -------
        generator
            Generator yielding (n-gram, frequency, probability)-tuples
            or (target, frequency, probability) tuples (if return_n_gram=True)
        """
        if not sizes:
            sizes = [self.n]

        for n_gram_string, frequency in self.completions():
            n_gram = n_gram_string.split(self.splitchar)

            if len(n_gram) not in sizes:
                continue

            target = n_gram[-1]
            if target in self.targets:
                probability = self._probability(n_gram)
                if return_n_gram:
                    yield n_gram, frequency, probability
                else:
                    yield target, frequency, probability

    def frequency(self, n_gram):
        """Return frequency of n_gram.

        Parameters
        ----------
        n_gram : list/tuple of str

        Returns
        -------
        int
            Frequency
        """
        n_gram_string = self.splitchar.join(n_gram)
        frequency = self._counts.frequency(n_gram_string)
        return frequency

    def completions(self, prefix=""):
        """Generator that returns all completions for a given prefix.

        Parameters
        ----------
        prefix : str
            Prefix that all results returned begin with.

        Yields
        -------
        Tuple
            Each complete n-gram with frequency as a (str, int)-tuple
        """
        if not self.must_contain:
            return self._counts.completions(prefix)

        for completion, frequency in self._counts.completions(prefix):
            completion_ = completion.split("#")
            if not any([word in self.must_contain for word in completion_]):
                continue

            yield completion, frequency

    def _train(self, n_gram):
        # test for OOV words
        for idx, word in enumerate(n_gram):
            if word not in self.vocabulary:
                n_gram = list(n_gram)[:idx]
                break

        # ensure n-gram contains target word if provided
        if self.must_contain:
            if not any([word in self.must_contain
                        for word in n_gram]):
                return

        n_gram_string = self.splitchar.join(n_gram)
        self._counts.insert(n_gram_string)

    def _probability(self, n_gram):
        frequency = self.frequency(n_gram)

        if frequency == 0:
            return 0

        *preceding, target = n_gram
        total = self.frequency(preceding)

        probability = frequency / total
        return probability

    def __contains__(self, n_gram):
        return n_gram in self._counts

    def __iter__(self):
        return self.completions()

    @property
    def n(self):
        return self._n

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def targets(self):
        return self._targets

    @property
    def must_contain(self):
        return self._must_contain

    @property
    def boundary(self):
        return self._boundary

    @property
    def splitchar(self):
        return self._splitchar


def train_lm(corpus, n,
             vocabulary=None, targets=None, must_contain=None,
             **kwargs):
    """Convenience function to train n-gram model on tagged corpus.
    """
    corpus = extract_fields(corpus, **kwargs)
    lm = LanguageModel(n,
                       vocabulary=vocabulary,
                       targets=targets,
                       must_contain=must_contain)
    lm.train(corpus)
    return lm
