from itertools import chain
from collections import Counter

from corpustools import extract_fields, ngrams
from language_model import LanguageModel

DUMMY_CORPUS = "dummy_corpus.txt"
DUMMY_SPECS = {"tag_field": 2,
               "delimiter": "\t",
               "num_fields": 3}

with open(DUMMY_CORPUS) as corpus:
    tokens = list(extract_fields(corpus, **DUMMY_SPECS))

dummy_counts = Counter(chain(ngrams(tokens, 1, join_char="#"),
                             ngrams(tokens, 2, join_char="#"),
                             ngrams(tokens, 3, join_char="#")))
dummy_counts[""] = len([t for t in tokens if not t.startswith("<")])


def test_frequencies_and_probabilities_of_trigram_model():
    lm = LanguageModel(3)
    lm.train(tokens)
    for result in lm.all_target_probabilities(return_n_gram=True,
                                              sizes=range(1, 4)):
        n_gram, frequency, probability = result
        *preceding, target = n_gram
        target_freq = dummy_counts["#".join(n_gram)]
        preceding_freq = dummy_counts["#".join(preceding)]
        target_prob = target_freq / preceding_freq
        assert frequency == target_freq
        assert probability == target_prob


def test_restricting_all_target_probabilities_to_size():
    lm = LanguageModel(3)
    lm.train(tokens)
    for token, freq, prob in lm.all_target_probabilities(sizes=[1]):
        assert freq == dummy_counts[token]
        assert prob == dummy_counts[token] / dummy_counts[""]


def test_all_words_included_in_language_model():
    lm = LanguageModel(3)
    lm.train(tokens)
    for token in tokens:
        if not token.startswith("<"):
            assert token in lm


def test_must_contain():
    lm = LanguageModel(3, must_contain={"this", "test"})
    lm.train(tokens)

    # test all target n-grams in lm have correct count
    for n_gram, freq, _ in lm.all_target_probabilities(return_n_gram=True):
        if "this" in n_gram or "test" in n_gram:
            print(n_gram)
            assert freq == dummy_counts["#".join(n_gram)]

    # test all target n-grams are contained in lm with correct counts
    for n_gram_string in dummy_counts:
        n_gram = n_gram_string.split("#")

        if "</s>" in n_gram_string:
            continue

        if any([w in {"this", "test"} for w in n_gram]):
            assert n_gram_string in lm
            assert lm.frequency(n_gram) == dummy_counts[n_gram_string]


def test_vocabulary_provided():
    pass


def test_targets_provided():
    pass
