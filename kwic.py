import io
import json
import math
from collections import OrderedDict

import configargparse
import numpy as np
import pandas as pd
import spacy


IGNORES = [
  'a.','b.','c.','d.','e.','f.','g.','h.','i.','j.','k.','l.','m.','n.','o.',
  'p.','q.','r.','s.','t.','u.','v.','w.','x.','y.','z.','aa.','bb.','cc.',
  'dd.','ee.', 'ff.',
]

# -----------------------------------------------------------------------------
# simple functions


def buildPipeline():
    # Load a previous trained-model
    nlp = spacy.load('en_core_web_sm')
    return nlp


def idf(word, doc_freq_vec, total_docs):
    # Fetch the number of documents that contain the word
    D = doc_freq_vec[word] if word in doc_freq_vec else 1

    # Compute inverse doc frequency
    # The rarer the word in general, the higher the score
    return math.log10(total_docs / D)


def removeStopwords(doc, ignores):
    for token in doc:
        if (
            not token.is_stop and
            not token.is_punct and
            not token.is_digit and
            not token.is_space and
            not token.text in ignores
        ):
            yield token

# -----------------------------------------------------------------------------
# Dataframe builders


def bagsOfWords(doc, known, options):
    # Load the SpaCy doc as a data frame (1 token per row)
    attrs = [
        [T.text, T.lemma_, T.pos_, 1]
        for T in removeStopwords(doc, options.ignores)
    ]
    df = pd.DataFrame(attrs, columns=['word', 'lemma', 'pos', 'count'])

    # Identify the words not in the known vocabulary
    df['known'] = df['lemma'].apply(lambda x: x in known)

    # Partiion the datasets
    df_known = df[df['known'] == True]
    df_unknown = df[df['known'] == False]

    # Create a pivot of lemma x part of speech
    reduced = df_known.groupby(['pos', 'lemma'])['count'].count().unstack(0)

    # Proper nouns may or may not be in the corpus
    propers = df.groupby(['pos', 'word'])['count'].count().unstack(0)

    # Make the panel
    bags = {
        'all_lemmas': df_known.groupby('lemma')['count'].count(),
        'propers': propers.loc[propers['PROPN'].notna(), 'PROPN'],
        'nouns': reduced.loc[reduced['NOUN'].notna(), 'NOUN'],
        'verbs': reduced.loc[reduced['VERB'].notna(), 'VERB'],
        'unknown': df_unknown.groupby('word')['count'].count()
    }

    # Fix the data sets
    #    1. Make sure the column is 'count' and not 'NOUN' or 'VERB'
    #    2. Change from a Series to a DataFrame
    #    3. Instead of "lemma' as the index/label, replace it with a ranged index
    for k, v in bags.items():
        bags[k] = v.rename('count').to_frame().reset_index()

    # Remove the proper nouns from unknown
    dups = bags['unknown'].merge(
        bags['propers'],
        on='word',
        how='left',
        indicator=True,
        suffixes=(None, '_y')
    )

    bags['unknown'] = dups[dups['_merge'] == 'left_only'].drop(
        columns=['_merge', 'count_y']
    )

    return bags


# -----------------------------------------------------------------------------
# DataFrame functions

def extractTopWords(df, keyCol, valueCol, topCount):
    x = df.sort_values(by=[valueCol], ascending=False)
    x.set_index(keyCol, inplace=True)
    return x[valueCol].iloc[:topCount].to_dict(OrderedDict)

# https://en.wikipedia.org/wiki/Tf%E2%80%93idf
def scoreTfidf(df, corpus, total_docs):
    total_words = df['count'].sum()
    df['tf'] = df['count'] # / total_word
    df['idf'] = np.vectorize(idf)(df['lemma'], corpus, total_docs)
    df['score'] = df['tf'] * df['idf']

# -----------------------------------------------------------------------------
# Sentence processing

def findImportantSentences(doc, nouns, verbs, propers, options):
    important = []
    for i, sent in enumerate(doc.sents):
        scan = set()
        for tok in sent:
            ntok = tok.lemma_.lower()
            if tok.text in propers:
                scan.add(tok.text + '@x')
            elif tok.pos_ == 'NOUN' and ntok in nouns:
                scan.add(ntok + '@n')
            elif tok.pos_ == 'VERB' and ntok in verbs:
                scan.add(ntok + '@v')

        if len(scan) >= options.min_occ:
            important.append((i, sent, scan))

    return important


def findClusters(sents, keyWords):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    # Translate the keywords array into O(1) lookup for position
    idx = {x: i for i, x in enumerate(keyWords)}

    # build adjacency matrix
    adjM = np.eye(len(keyWords), dtype=int)
    for _, _, scan in sents:
        queue = list(scan)
        while queue != []:
            a = queue.pop(0)
            ia = idx[a]
            for b in queue:
                ib = idx[b]
                adjM[ia, ib] += 1
                adjM[ib, ia] += 1

    # Turn the adjacency matrix into a graph
    graph = csr_matrix(adjM)

    # Reduce to connected components
    n_components, labels = connected_components(
        csgraph=graph, directed=False, return_labels=True
    )

    # Process into a useable format
    groups = []
    for i in range(n_components):
        groups.append([
            keyWords[j].split('@')[0]
            for j, g in enumerate(labels)
            if i == g
        ])

    return groups


def markupSentencesCli(sents, nouns, verbs, propers, options):
    def textPropers(s):
        return u'\u001b[4m{}\u001b[0m'.format(s)

    def textNoun(s):
        return u'\u001b[32m{}\u001b[0m'.format(s)

    def textVerb(s):
        return u'\u001b[33m{}\u001b[0m'.format(s)

    newline = ''
    if options.handle_newline == 'space':
        newline = ' '

    for i, sent, _ in sents:
        s = ''
        for tok in sent:
            if tok.text in propers:
                s += textPropers(tok.text) + tok.whitespace_
            elif tok.pos_ == 'NOUN' and tok.lemma_.lower() in nouns:
                s += textNoun(tok.text_with_ws)
            elif tok.pos_ == 'VERB' and tok.lemma_.lower() in verbs:
                s += textVerb(tok.text_with_ws)
            elif tok.is_space and '\n' in tok.text:
                s += tok.text if options.handle_newline == 'keep' else newline
            else:
                s += tok.text_with_ws

        print(f'[{i}] {s}')

# -----------------------------------------------------------------------------
# Display

def outputStats(doc, sents, nouns, verbs, propers, options):
    print('\nStats')
    total_sents = len(list(doc.sents))
    print(f'\t{total_sents} sentences')
    print(f'\t{len(sents)} important')

    def safeNext(iterator):
        try:
            k, f = next(iterator)
        except StopIteration:
            k = ''
            f = 0.0
        return k, f

    # iterators
    inoun = iter(nouns.items())
    iverb = iter(verbs.items())
    iproper = iter(propers.items())

    fmt_head = '    {:^23}   {:^23}   {:^23}'
    fmt_row = '{:>2d}: {:<19} {:>3.0f}   {:<19} {:>3.0f}   {:<19} {:>3.0f}'

    print('\n')
    print(fmt_head.format('Proper Nouns/Acronyms', 'Nouns', 'Verbs'))
    print(fmt_head.format('=' * 23, '=' * 23, '=' * 23))
    for i in range(options.top):
        pk, pv = safeNext(iproper)
        nk, nv = safeNext(inoun)
        vk, vv = safeNext(iverb)

        print(fmt_row.format(
            i + 1,
            pk, pv,
            nk, nv,
            vk, vv
        ))

# -----------------------------------------------------------------------------
# Main

def build_arg_parser():
    p = configargparse.ArgParser(prog='kwic',
                                 description='shows key words in context')
    g = p.add_argument_group('Analysis')
    g.add('--top', type=int, default=25,
          help='how many high scoring nouns and verbs are keywords')
    g.add('--min-occ', type=int, default=3,
          help='number of key words in an important sentence')
    g = p.add_argument_group('Display')
    g.add('--handle-newline', default='ignore',
          choices=['keep', 'ignore', 'space'],
          help='decide how newlines should be handled')
    g = p.add_argument_group('I/O')
    g.add('document',
          help='the file to analyze')

    return p


def main():
    # Get the arguments from the command line
    p = build_arg_parser()
    options = p.parse_args()
    setattr(options, 'ignores', IGNORES)

    print('Reading Corpus of Contemporary American English...')
    with io.open('corpus.json','r', encoding='utf-8') as f:
        coca = json.load(f)

    # Open a text file
    print(f'Reading {options.document}...')
    with io.open(options.document,'r', encoding='utf-8') as f:
        raw = f.read()

    # Make the bag of words
    print('Analyzing document in spaCy...')
    nlp = buildPipeline()
    doc = nlp(raw)

    print('Partioning NLP results...')
    bags = bagsOfWords(doc, coca['all_lemmas'], options)

    print('Scoring the document...')
    for subset in ['nouns', 'verbs']:
        scoreTfidf(bags[subset], coca[subset], coca['total_docs'])

    print('Extracting the key words')
    unknowns = extractTopWords(bags['unknown'], 'word', 'count', 4000)
    nouns = extractTopWords(bags['nouns'], 'lemma', 'score', options.top)
    verbs = extractTopWords(bags['verbs'], 'lemma', 'score', options.top)
    propers = extractTopWords(bags['propers'], 'word', 'count', options.top)

    print('\nKey words in context')
    sents = findImportantSentences(doc, nouns, verbs, propers, options)

    # keyWords = [x + '@n' for x in nouns]
    # keyWords.extend([x + '@v' for x in verbs])
    # keyWords.extend([x + '@x' for x in propers])
    # groups = findClusters(sents, keyWords)
    # print(groups)

    markupSentencesCli(sents, nouns, verbs, propers, options)
    outputStats(doc, sents, nouns, verbs, propers, options)


if __name__ == '__main__':
    main()
