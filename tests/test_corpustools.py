import gzip
import tempfile

from os.path import dirname, join
from collections import Counter

from corpustools import ENGLISH
from corpustools import add_most_frequent
from corpustools import ContainsEverything
from corpustools import merge_tokens_tags_corpus, replace_disallowed
from corpustools import extract_units, extract_fields, split_collection
from corpustools import filter_tagged_vocabulary, filter_tagged_event_file


top = join(dirname(__file__), "data")

DUMMY_CORPUS = join(top, "dummy_corpus.txt")
DUMMY_MERGED = join(top, "dummy_corpus_merged.txt")
DUMMY_EVENTS = join(top, "dummy_corpus_merged_events.gz")
DUMMY_EVENTS_FILTERED = join(top, "dummy_corpus_merged_events_filtered.gz")
DUMMY_EVENTS_FILLED = join(top, "dummy_corpus_merged_events_filtered_fill_cues.gz")

DUMMY_SPECS = {"tag_field": 2,
               "delimiter": "\t",
               "num_fields": 3}


def test_extract_tokens():
    with open(DUMMY_CORPUS) as corpus:
        tokens = extract_fields(corpus,
                                keep_meta={},
                                **DUMMY_SPECS)
        tokens = list(tokens)
        assert len(tokens) == 38


def test_extract_tags_tokens():
    with open(DUMMY_CORPUS) as corpus:
        tags_tokens = extract_fields(corpus,
                                     keep_meta={},
                                     return_fields={1, 2},
                                     **DUMMY_SPECS)
        tags_tokens = list(tags_tokens)
        assert len(tags_tokens) == 38
        assert all([len(tt) == 2 for tt in tags_tokens])


def test_extract_sentences():
    with open(DUMMY_CORPUS) as corpus:
        sentences = extract_units(corpus,
                                  **DUMMY_SPECS)
        sentences = list(sentences)
        assert len(sentences) == 3


def test_extract_documents():
    with open(DUMMY_CORPUS) as corpus:
        documents = extract_units(corpus,
                                  boundary="</doc>",
                                  **DUMMY_SPECS)
        documents = list(documents)
        assert len(documents) == 2


def test_split_documents_into_sentences():
    with open(DUMMY_CORPUS) as corpus:
        documents = extract_units(corpus,
                                  boundary="</doc>",
                                  keep_meta={"</doc>", "</s>"},
                                  **DUMMY_SPECS)
        documents = list(documents)
        sentences = [sent for document in documents
                     for sent in split_collection(document, "</s>")]
        assert len(sentences) == 3


def test_merge_tokens_tags_corpus():
    with tempfile.NamedTemporaryFile() as tmp:
        merge_tokens_tags_corpus(DUMMY_CORPUS, tmp.name,
                                 symbols=ENGLISH,
                                 overwrite=True,
                                 replacement="repl",
                                 **DUMMY_SPECS)
        with open(tmp.name) as test, open(DUMMY_MERGED) as standard:
            for line in test:
                assert line == standard.readline()


def test_replace_disallowed_tokens():
    sequence = ["the", "last", "token", "contains",
                "a", "disallowed", "character", "test-word"]
    replaced = replace_disallowed(sequence, ENGLISH, "repl")
    assert sequence[:-1] == replaced[:-1]
    assert replaced[-1] == "repl"


def test_replace_disallowed_fields():
    sequence = [["the", "dt"], ["next", "adj"], ["token", "nn"],
                ["is", "vb"], ["disallowed", "adj"], ["test-word", "nn"],
                ["as", "cc"], ["is", "vb"], ["the", "dt"], ["next", "adj"],
                ["item", "nn"], ["!", "$"]]
    replaced = replace_disallowed(sequence, ENGLISH, "repl")
    assert sequence[:5] + sequence[6:11] == replaced[:5] + replaced[6:11]
    assert ["repl", sequence[5][1]] == replaced[5]
    assert ["repl", "repl"] == replaced[11]


def test_filter_tagged_vocabulary():
    tagged_vocabulary = {"test|nn", "test|vb", "the|dt",
                         "is|vb", "this|dt"}
    vocabulary = {"test", "this"}
    target = {"test|nn", "test|vb", "this|dt"}
    filtered = filter_tagged_vocabulary(tagged_vocabulary, vocabulary, "|")
    assert filtered == target


def test_add_most_frequent():
    targets = {"apple", "banana", "orange", "dragonfruit"}
    vocab = Counter({"apple": 10, "banana": 20, "dragonfruit": 30,
                     "mango": 40, "kiwi": 50, "pear": 60})
    added = add_most_frequent(targets, vocab, 6)
    expected = {"apple", "banana", "orange", "dragonfruit",
                "kiwi", "pear"}
    assert added == expected


def test_add_most_frequent_filter_targets():
    targets = {"apple", "banana", "orange", "dragonfruit"}
    vocab = Counter({"apple": 10, "banana": 20, "dragonfruit": 30,
                     "mango": 40, "kiwi": 50, "pear": 60})
    added = add_most_frequent(targets, vocab, 6, filter_targets=True)
    expected = {"apple", "banana", "dragonfruit",
                "mango", "kiwi", "pear"}
    assert added == expected


def test_contains_everything():
    container = ContainsEverything()
    assert "test" in container


def test_filter_tagged_event_file():
    cues = {"code", "functions", "sentence", "symbol"}
    outcomes = {"a", "the"}
    with tempfile.NamedTemporaryFile() as tmp:
        filter_tagged_event_file(DUMMY_EVENTS,
                                 tmp.name,
                                 cues=cues,
                                 outcomes=outcomes,
                                 overwrite=True)
        with gzip.open(DUMMY_EVENTS_FILTERED, "rt") as target, \
                gzip.open(tmp.name, "rt") as test:
            for line in target:
                cues, *outcome = line.strip().split("\t")
                cues = cues.split("_")
                test_line = test.readline()
                test_cues, *test_outcome = test_line.strip().split("\t")
                test_cues = test_cues.split("_")
                assert set(test_cues) == set(cues)
                assert test_outcome == outcome


def test_filter_tagged_event_file_fill_cues():
    cues = {"code", "functions", "sentence", "symbol"}
    outcomes = {"a", "the"}
    with tempfile.NamedTemporaryFile() as tmp:
        filter_tagged_event_file(DUMMY_EVENTS,
                                 tmp.name,
                                 cues=cues,
                                 outcomes=outcomes,
                                 fill_cues=5,
                                 overwrite=True)
        with gzip.open(DUMMY_EVENTS_FILLED, "rt") as target, \
                gzip.open(tmp.name, "rt") as test:
            for line in target:
                cues, *outcome = line.strip().split("\t")
                cues = cues.split("_")
                test_line = test.readline()
                test_cues, *test_outcome = test_line.strip().split("\t")
                test_cues = test_cues.split("_")
                assert set(test_cues) == set(cues)
                assert test_outcome == outcome
