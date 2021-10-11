#!/usr/bin/env python3
# Copyright 2020 Maria Cervera
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# title           :data/timeseries/preprocess_mud.py
# author          :mc
# contact         :mariacer@ethz.ch
# created         :09/08/2020
# version         :1.0
# python_version  :3.7
"""
Script to structure the multilingual universal dependencies dataset, which
can then be used via :class:`data.timeseries.mud_data.MUDData`.

The result of this script is available at

    https://www.dropbox.com/s/fqo44xvxhx2br1t/mud_data.pickle?dl=0

If you want to recreate or modify this dataset, download verison 2.6 of the
UD data set from

    https://universaldependencies.org/#download

Since different tree banks exist for each language, we selected those with
the largest number of tokens (above 60k) that use the 17 universal tags.
The resulting tree banks for each language obtained from version v2.6 are:
    ar ARABIC: PADT
    cs CZECH: PDT
    da DANISH: DDT
    de GERMAN: GSD
    en ENGLISH: GUM
    es SPANISH: AnCora
    fi FINNISH: FTB
    fr FRENCH: GSD
    he HEBREW: HTB
    hi HINDI: HDTB
    hr CROATIAN: SET
    id INDONESIAN: GSD
    it ITALIAN: ISDT
    lt LITHUANIAN: ALKSINS
    nl DUTCH: ALPINO
    no NORWEGIAN: BOKMAAL
    pl POLISH: PDB
    pt PORTUGESE: Bosque
    sl SLOVENE: SSJ
    sv SWEDISH: LinES

The corresponding folders should be located in:

    ``../../datasets/sequential/mud/universal_dependencies``

Equivalently, the word embeddings from each language can be obtained from:

    https://sites.google.com/site/rmyeid/projects/polyglot?authuser=0

and the corresponding pickle files for each language should be located in:

    ``../../datasets/sequential/mud/word_embeddings``
"""
import numpy as np
import pickle
import os
from warnings import warn
from conllu import parse_incr
import re

def words2integers(words_sentence, words, normalize_words=True):
    """Transform list of words in a sentence into a list of corresponding
    integers.

    Args:
        sentence (list): The sentence words.
        words (list): The vocabulary.
        normalize_words (bool, optional): If True, unknown words will be tried
            to be matched with existing words.

    Returns:
        (list): The list of integers.
    """
    integers_list = []
    for token in words_sentence:

        # If the index is not in the vocabulary, try to find it.
        if token not in words and normalize_words:
            token = normalize(token, words)
        # If it is still not in the vocabulary, return the index of the '<UNK>'
        # word corresponding to unknown words.
        if token not in words:
            idx = 0
        else:
            idx = words.index(token)
        integers_list.append(idx)

    return integers_list


def case_normalizer(word, dictionary):
    """In case the word is not available in the vocabulary, we can try multiple
    case normalizing procedures. We consider the best substitute to be the one
    with the lowest index, which is equivalent to the most frequent
    alternative.

    Obtained from

        https://nbviewer.jupyter.org/gist/aboSamoor/6046170

    Args:
        word (str): The word.
        dictionary (list): The dictionary.

    Returns:
        (str): The case-normalized word.

    """
    w = word
    lower = (dictionary.get(w.lower(), 1e12), w.lower())
    upper = (dictionary.get(w.upper(), 1e12), w.upper())
    title = (dictionary.get(w.title(), 1e12), w.title())
    results = [lower, upper, title]
    results.sort()
    index, w = results[0]
    if index != 1e12:
        return w
    return word


def normalize(word, dictionary):
    """Find the closest alternative in case the word is out of vocabulary.

    Numbers will be transformed into the '#' character, and unknown words will
    be case normalized to see if they match some word of the vocabulary. If
    not, a ``None`` will be returned.

    Obtained from

        https://nbviewer.jupyter.org/gist/aboSamoor/6046170

    Args:
        word (str): The word.
        dictionary (int): The dictionary.

    Returns:
        (str): The closest word.
    """
    DIGITS = re.compile("[0-9]", re.UNICODE)
    word_id = {w:i for (i, w) in enumerate(dictionary)}
    if not word in word_id:
        word = DIGITS.sub("#", word)
    if not word in word_id:
        word = case_normalizer(word, word_id)

    if not word in word_id:
        return None
    return word


def tags2labels(tags_sentence, tagset):
    """Transform list of tags in a sentence into list of labels of the tags.

    Args:
        tags_sentence (list): The sentence tags.
        tagset (list): The tagset.

    Returns:
        (list): The list of labels.
    """
    labels = []
    for tag in tags_sentence:
        if tag in tagset:
            idx = tagset.index(tag)
        # A '_' tag corresponds to words that have been split and thus
        # duplicated and will be removed later.
        elif tag == '_':
            idx = '_'
        else:
            raise ValueError
        labels.append(idx)

    return labels


def extract_sentences(path, filenames, split='train'):
    """Extract word-tag lists of sentences from downloaded conllu files.

    Args:
        path (str): The path to the stored data.
        filenames (str): The name of the files for the current treebank.
        split (optional, str): The split to be returned:`train`,`test` or `dev`.

    Returns:
        (list): A list of length equal to the number of sentences, where each
        element is a dict containing two lists: one with the words and
        one with the corresponding tags.
    """
    path = os.path.join(path, '%s%s.conllu'%(filenames, split))
    data_file = open(path, "r", encoding="utf-8")

    sentences = []
    for tokenlist in parse_incr(data_file):
        words = []
        tags = []
        for t in list(tokenlist):
            words.append(t['form'])
            tags.append(t['upos'])
        sentences.append({'words': words, 'udtag': tags})

    data_file.close()

    return sentences


def preprocess_language(data, words, tagset, max_num_tokens=300000,
        max_sentence_len=None):
    """Preprocess the data from a given split of given language.

    This function turns the set of sentences in the current split into lists of
    integers (for the words) and one-hot encoded labels (for the tags).

    Furthermore, this function gets rid of word-tag pairs of words that are
    split and duplicated. For example, the italian word "dalla" is split to
    "da" and "la", and all three words appear in the provided sentences together
    with the two tags of "da" and "la", and a "_" tag for the original word
    "dalla".

    Args:
        data (list): The list of sentences, where each item is a tuple with
            the list of words, and the list of tags in a given sentence.
        words (list): The vocabulary of the current dataset.
        tagset (list): The set of tags.
        max_num_tokens (int, optional): The maximum number of tokens.
        max_sentence_len (int, optional): The maximum length of sequence to be
            considered. Will be used as length up to which sentences need to
            be padded.

    Returns:
        (tuple): Tuple containing:

        - **x**: The inputs (words encoded as integers).
        - **y**: The outputs (tags encoded as one-hot).
        - **sentence_lens**: The lengths of the sentences before padding.
    """

    ### Turn the sentences into lists of integers and tags.
    # Each integer value is the index of a certain word within the
    # vocabulary, and each tag value is one of 17 universal tags shared
    # accross languages.
    all_integers = []
    all_labels = []
    num_tokens = 0
    for s, sentence in enumerate(data):
        if max_num_tokens != -1 and num_tokens >= max_num_tokens:
            break
        integers = words2integers(sentence['words'], words)
        labels = tags2labels(sentence['udtag'], tagset)

         #Make sure that no '_' tags are present. These occur whenever a certain
         #word is split into two lemmas, e.g. alla --> ['alla', 'al', 'la']
         #so we can get rid of that word and the corresponding tag '_'.
        if '_' in labels:
            idxs_to_remove = [i for i, x in enumerate(labels) if x == '_']
            # Remove the elements from both lists.
            labels = [t for i, t in enumerate(labels) if i not in idxs_to_remove]
            integers = [t for i, t in enumerate(integers) if i not in \
                idxs_to_remove]
        assert '_' not in labels
        assert len(labels) == len(integers)

        # If the sentence contains a None, i.e. a word that couldn't be
        # found within the vocabulary, we discard the entire sentence.
        if not 0 in integers:
            all_integers.append(integers)
            all_labels.append(labels)
            num_tokens += len(integers)

    num_sentences = len(all_labels)

    ### Turn the list of integers and labels into padded arrays.
    # We use a -1 zero pad to easily recover sequence length later.
    if max_sentence_len is None:
        max_sentence_len = np.max([len(integers) for integers in all_integers])
    x = np.zeros((num_sentences, max_sentence_len))
    y = np.zeros((num_sentences, max_sentence_len))
    sentence_lens = np.zeros(num_sentences)
    for i, (ints, labels) in enumerate(zip(all_integers, all_labels)):
        sentence_lens[i] = len(ints)
        x[i, :len(ints)] = ints
        y[i, :len(ints)] = labels
    assert len(sentence_lens) == num_sentences

    return x, y, sentence_lens


if __name__=='__main__':
    warn('The script was created for one time usage and has to be adapted ' +
         'when reusing it.')

    download_dir = '../../datasets/sequential/mud'
    target_path = '../../datasets/sequential/mud'

    if not os.path.exists(download_dir):
        raise RuntimeError('Pathnames have to be adapted manually before ' +
                           'using the script.')

    # Hardcode some data namings.
    languages = ['en', 'ar', 'cs', 'da', 'de', 'es', 'fi', 'fr', 'he', 'hi',
        'hr', 'id', 'it', 'lt', 'nl', 'no', 'pl', 'pt', 'sl', 'sv']
    foldernames = ['UD_English-GUM', 'UD_Arabic-PADT', 'UD_Czech-PDT',
        'UD_Danish-DDT', 'UD_German-GSD', 'UD_Spanish-AnCora',
        'UD_Finnish-FTB', 'UD_French-GSD', 'UD_Hebrew-HTB',
        'UD_Hindi-HDTB', 'UD_Croatian-SET', 'UD_Indonesian-GSD',
        'UD_Italian-ISDT', 'UD_Lithuanian-ALKSNIS', 'UD_Dutch-Alpino',
        'UD_Norwegian-Bokmaal', 'UD_Polish-PDB', 'UD_Portuguese-Bosque',
        'UD_Slovenian-SSJ', 'UD_Swedish-LinES']
    filenames = ['en_gum-ud-', 'ar_padt-ud-', 'cs_pdt-ud-', 'da_ddt-ud-',
        'de_gsd-ud-', 'es_ancora-ud-', 'fi_ftb-ud-', 'fr_gsd-ud-',
        'he_htb-ud-', 'hi_hdtb-ud-', 'hr_set-ud-', 'id_gsd-ud-',
        'it_isdt-ud-', 'lt_alksnis-ud-', 'nl_alpino-ud-', 'no_bokmaal-ud-',
        'pl_pdb-ud-', 'pt_bosque-ud-', 'sl_ssj-ud-', 'sv_lines-ud-']

    # Code to reduce set of languages.
    use_reduced_set = False
    if use_reduced_set:
        languages = ['en', 'de', 'fr', 'pt', 'it']
        foldernames = ['UD_English-GUM', 'UD_German-GSD', 'UD_French-GSD',
            'UD_Portuguese-Bosque', 'UD_Italian-ISDT']
        filenames = ['en_gum-ud-', 'de_gsd-ud-', 'fr_gsd-ud-', 'pt_bosque-ud-', \
            'it_isdt-ud-']

    num_languages = len(languages)


    ### get tagset with 17 tags from english treebank
    # The tagset is shared accross languages.
    ud_path = os.path.join(download_dir,
                           'universal_dependencies/UD_English-GUM')
    fname = 'en_gum-ud-'
    train_sentences = extract_sentences(ud_path,fname,split='train')
    tagset = []
    for sentence in train_sentences:
        tags_sentence = sentence['udtag']
        tagset.extend(tags_sentence)
    tagset = list(set(tagset))
    num_tags = len(tagset)
    assert num_tags == 17


    ### Create list of datasets for each language.
    all_data = []
    all_embeddings = []
    all_dicts = []
    for l in range(num_languages):

        language = languages[l]
        print('\nProcessing language: "%s" (%i/%i)'%(language, \
            l+1,num_languages))

        ### Load the word embeddings of the current language.
        embeddings_file = os.path.join(download_dir, \
            'word_embeddings/polyglot-%s.pkl'%language)
        with open(embeddings_file, 'rb') as f:
            words, embeddings = pickle.load(f, encoding='bytes')

        ### Load the dataset of the current language.
        ud_path = os.path.join(download_dir, \
            'universal_dependencies/%s'%foldernames[l])
        train_sentences = extract_sentences(ud_path, filenames[l],split='train')
        test_sentences = extract_sentences(ud_path, filenames[l], split='test')
        val_sentences = extract_sentences(ud_path, filenames[l], split='dev')

        # Figure out the size of the longest sentence overall.
        max_len_train = np.max([len(s['words']) for s in train_sentences])
        max_len_test = np.max([len(s['words']) for s in test_sentences])
        max_len_val = np.max([len(s['words']) for s in val_sentences])
        max_sentence_len = np.max([max_len_train, max_len_test, max_len_val])

        ### Preprocess the sentences of each data split.
        train_dataset = preprocess_language(train_sentences, words, tagset,
            max_sentence_len=max_sentence_len)
        test_dataset = preprocess_language(test_sentences, words, tagset,
            max_sentence_len=max_sentence_len)
        val_dataset = preprocess_language(val_sentences, words, tagset,
            max_sentence_len=max_sentence_len)

        all_data.append([train_dataset, test_dataset, val_dataset])
        all_embeddings.append(embeddings)
        all_dicts.append(words)

        print('%i sentences are considered.'%\
            (len(train_dataset[0])+len(test_dataset[0])+len(val_dataset[0])))

    ### Save everything.
    with open(os.path.join(target_path, 'mud_data.pickle'), 'wb') as f:
            pickle.dump([all_data, all_dicts, tagset], f)
    with open(os.path.join(target_path, 'embeddings.pickle'), 'wb') as f:
            pickle.dump(all_embeddings, f)
    print('Processing done. Data from %i language(s) has been pickled.'\
        %num_languages)

