import os
import re
import sys
import random
import warnings
from os.path import exists

import psutil
from pyndl.count import cues_outcomes
from pyndl.preprocess import filter_event_file

POLISH_LOWER = "aąbcćdeęfghijklłmnńoóprsśtuwyzźżqvx"
POLISH_UPPER = POLISH_LOWER.upper()
POLISH = POLISH_UPPER + POLISH_LOWER

ENGLISH_LOWER = "abcdefghijklmnopqrstuvwxyz"
ENGLISH_UPPER = ENGLISH_LOWER.upper()
ENGLISH = ENGLISH_LOWER + ENGLISH_UPPER


def extract_fields(corpus,
                   delimiter="\t",
                   lower=True,
                   drop_meta=True,
                   keep_meta={"</s>"},
                   drop_tags={"zz", "sy"},
                   tag_field=2,
                   num_fields=5,
                   return_fields=0
                   ):
    """Generator that filters lines and extracts fields from tagged corpus.

    Parameters
    ----------
    corpus : iterable of str
        Corpus text stream, typically an opened file to read from
    delimiter : str
        String that delimits fields (token, tag, ..) in lines in corpus
    lower : bool
        Treat all characters as lowercase
    drop_meta : bool
        Drop meta tags, i.e. lines with only one field starting with "<"
    keep_meta : collection
        If drop_meta==True, these lines (e.g. sentence boundaries) will
        be kept.
    drop_tags : collection
        Lines with any of these tags (e.g. punctuation, symbols) will
        be dropped.
    tag_field : int
        Field where the tag can be found (for dropping)
    num_fields : int
        Number of fields to split each line into. If lines with fewer
        fields are encountered, a warning is issued.
    return_fields : int or list of int
        Which fields to return. If integer is given, only that field will
        be returned, otherwise a list of fields will be returned.

    Yields
    -------
    str or list of str
        Field(s) - String if type(num_fields) is int, list of str otherwise.
    """
    for idx, line_ in enumerate(corpus):

        line = line_.rstrip("\n")

        if lower:
            line = line.lower()

        if not line:
            continue

        if line in keep_meta:
            yield line
            continue

        fields = line.split(delimiter)

        if len(fields) == 1 and drop_meta and line.startswith("<"):
            continue

        if len(fields) < num_fields:
            msg = f"Line ({idx}) with fewer elements than {num_fields} \
                    ({len(fields)} encountered:\n{line_}"
            warnings.warn(msg)
            continue

        if drop_tags:
            if fields[tag_field] in drop_tags:
                continue

        if type(return_fields) == int:
            yield fields[return_fields]
        else:
            yield [fields[idx] for idx in return_fields]


def extract_units(corpus,
                  boundary="</s>",
                  **kwargs):
    """Generator that yields units (e.g. sentences) from corpus.

    Parameters
    ----------
    corpus : iterable of str
        Sequence of str to break into units, typically lines from
        a tagged corpus
    boundary : str
        String that separates units,
        e.g. meta tag </s> for sentences (default)

    Yields
    -------
    list of str or list of lists
        each unit as an iterable of tokens or fields

    Notes
    -----
    Other keyword arguments are passed on to extract_fields.
    In particular keep_meta must be specified if meta tags are
    to be retained and return_fields if fields other than
    the default (0 for token) are to be extracted.
    """
    if "keep_meta" not in kwargs:
        kwargs["keep_meta"] = {boundary}

    corpus = extract_fields(corpus, **kwargs)
    current_unit = list()

    for token in corpus:
        if token == boundary:
            yield current_unit
            current_unit = list()
            continue

        current_unit.append(token)

    if current_unit:
        return current_unit


def replace_disallowed(sequence, symbols, replacement):
    """Replace tokens or fields in a sequence based on symbols they contain.

    Parameters
    ----------
    sequence : collection of str or collections
        Sequence of tokens to be replaced.
    symbols : str
        Symbols that are allowed. Tokens containing other symbols are replaced.
    replacement : object
        Object (typically string) that illegal tokens are replaced with

    Notes
    -----
    If token is not string, but a collection of strings, each element
    in the collection will be replaced if it contains disallowed symbols
    """
    disallowed_characters = re.compile(f"[^{symbols}]")
    replaced = list()

    for token in sequence:
        if isinstance(token, str):
            if disallowed_characters.search(token):
                token = replacement

        else:
            token = [field if not disallowed_characters.search(field)
                     else replacement
                     for field in token]

        replaced.append(token)

    return replaced


def split_collection(collection, split):
    """Split collection on value similar to str.split().

    Parameters
    ----------
    collection : collection
        Collection to split
    split : object
        Value to split collection on

    Yields
    ------
    list
        Each subcollection after splitting
    """
    current = list()
    for element in collection:

        if element == split:
            if collection:
                yield current
            current = list()
            continue

        current.append(element)


def merge_tokens_tags_corpus(corpus_path, merged_corpus_path,
                             symbols=POLISH,
                             replacement="REPL",
                             token_field=0, tag_field=2,
                             overwrite=False,
                             **kwargs):
    """Turns tagged corpus (one token per line) into sentences
    with token and tag merged (one sentence per line).

    Parameters
    ----------
    corpus_path : str or path
        Path to tagged corpus file
    merged_corpus_path : str or path
        Path to resulting corpus file
    symbols : str
        string of symbols allowed in token and tag
    replacement : string
        String that illegal tokens/tags are replaced with
    token_field : int
        Field where token is located in corpus lines
    tag_field : int
        Field where tag is located in corpus lines
    overwrite : bool
        Overwrite merged_corpus_path if exists

    Notes
    -----
    Other keyword arguments are passed on to extract_units
    """
    if "|" not in symbols:
        symbols = symbols + "|"

    if exists(merged_corpus_path) and not overwrite:
        msg = f"'{merged_corpus_path}' already exists and overwrite=False!"
        raise OSError(msg)

    with open(corpus_path) as corpus:
        token_tag = [token_field, tag_field]
        sentences = extract_units(corpus=corpus,
                                  return_fields=token_tag,
                                  **kwargs)

        with open(merged_corpus_path, "wt") as merged:
            for sentence in sentences:
                sentence = replace_disallowed(sequence=sentence,
                                              symbols=symbols,
                                              replacement=replacement)
                sentence = ["|".join(fields) for fields in sentence]
                line = " ".join(sentence) + "\n"
                merged.write(line)


def filter_tagged_vocabulary(tagged_vocabulary, vocabulary, split="|"):
    """Filters tagged_vocabulary (tokens merged with tags) for tokens
    occurring in vocabulary.

    Parameters
    ----------
    tagged_vocabulary : collection
        vocabulary of tokens merged with tags
    vocabulary : collection
        target vocabulary of tokens without tags
    split : str
        string delimiting tags and tokens in tagged_vocabulary
    """
    targets = set()
    for tagged_word in tagged_vocabulary:
        word, tag = tagged_word.split(split)
        if word in vocabulary:
            targets.add(tagged_word)
    return targets


def add_most_frequent(targets, vocab, target_size, filter_targets=False):
    """Creates vocabulary of target_size from targets and most frequent
    words in vocab.

    Parameters
    ----------
    targets : collection
        List of targets to be included
    vocab : Counter
        Vocabulary to add most frequent words from
    target_size : int
        Size of vocabulary to be returned
    filter_targets : bool
        If true, targets that are not included in vocab are removed.
    """
    if filter_targets:
        targets = {target for target in targets if target in vocab}

    if len(targets) >= target_size:
        msg = "Size of targets larger than target_size!\n"
        warnings.warn(msg)
        return targets

    num = target_size - len(set(targets))
    vocab = [k for k, v in vocab.most_common(num) if k not in targets]
    targets.update(vocab)
    return targets


def filter_tagged_event_file(input_event_file,
                             filtered_event_file,
                             cues, outcomes,
                             fill_cues=0,
                             fill_outcomes=0,
                             overwrite=False,
                             number_of_processes=1):
    """Filters event file with tokens and tags merged for collections of
    untagged cues and outcomes.

    Parameters
    ----------
    input_event_file : str or path
        Path to event file with tokens and tags merged
    filtered_event_file : str or path
        Path to resulting event file
    cues : collection
        Collection of target cues
    outcomes : collection
        Collection of taret outcomes
    fill_cues : int
        Fill cues with most frequent words to size fill_cues.
        If 0, no words will be added.
    fill_outcomes : int
        Fill outcomes with most frequent words to size fill_outcomes.
        If 0, no words will be added.
    overwrite : bool
        Overwrite filtered_event_path if exists
    number_of_processes : int
        Number of processes to use
    """
    if exists(filtered_event_file) and not overwrite:
        msg = f"'{filtered_event_file}' already exists and overwrite=False!"
        raise OSError(msg)

    _, _, vocabulary = cues_outcomes(input_event_file,
                                     number_of_processes=number_of_processes)
    cues = filter_tagged_vocabulary(vocabulary, cues)
    outcomes = filter_tagged_vocabulary(vocabulary, outcomes)

    if fill_cues:
        cues = add_most_frequent(cues, vocabulary, fill_cues)

    if fill_outcomes:
        outcomes = add_most_frequent(outcomes, vocabulary, fill_outcomes)

    filter_event_file(input_event_file, filtered_event_file,
                      keep_cues=cues, keep_outcomes=outcomes,
                      number_of_processes=number_of_processes)


def ngrams(sequence, n, as_string=True, join_char=" "):
    """Extracts all n-grams of length n from sequence.

    Parameters
    ----------
    sequence : Sliceable container
        Sequence to extract n-grams from.
        Typically a string or sequence of strings
    n : int
        Size of n-grams to extract
    as_string : bool
        Return n-gram as single strings with spaces between tokens
    """
    for i in range(len(sequence) - n + 1):
        if as_string:
            yield join_char.join(sequence[i:i + n])
        else:
            yield sequence[i:i + n]


def random_strings(num_strings, symbols=ENGLISH,
                   min_len=1, max_len=15, seed=None):
    if seed:
        random.seed(seed)

    for i in range(num_strings):
        length = random.choice(range(min_len, max_len))
        yield random_string(length=length, symbols=symbols)


def random_string(length, symbols=ENGLISH, seed=None):
    if seed:
        random.seed(seed)

    return "".join(random.choices(symbols, k=length))


def verbose_generator(sequence, target,
                      every_n=1000, total="?", template=None,
                      text_buffer=sys.stdout):
    """Yields elements from sequence, counts target occurrences and writes
    progress.

    Parameters
    ----------
    sequence : iterable
        Generator to add verbosity to
    target : object
        Object in sequence to count
    every_n : int
        Write progress to text_buffer every every_n occurrence of target
    total : int
        Total number target will appear with
    template : str
        Template for the verbose message
    text_buffer : buffer
        Reports will be written by calling text_buffer.write() method

    Notes
    -----
    Template will be formatted with .format(), injecting:
        total - provided
        target - provided
        count - count of target
        memory - total memory usage of process using generator
    """
    if not template:
        template = "\rConsumed {count} {target} out of {total}."

    text_buffer.write("\n")
    count = 0

    for element in sequence:
        yield element

        if element == target:
            count += 1

        if not count % every_n:
            memory = memory_usage()
            msg = template.format(count=count, total=total,
                                  target=target, memory=memory)
            text_buffer.write(msg)
            text_buffer.flush()

    msg = template.format(count=count, total=total,
                          target=target, memory=memory)
    text_buffer.write(msg)
    text_buffer.write("\n")
    text_buffer.flush()


def memory_usage():
    """Returns total memory usage of current process in MB.
    """
    pid = os.getpid()
    p = psutil.Process(pid)
    memory = p.memory_full_info().uss / 1024 / 1024
    return memory


class ContainsEverything:
    """Dummy container that mimics containing everything.
    Has .add() method to mimic set.
    """

    def __contains__(self, _):
        return True

    def add(self, _):
        pass
