import io
import json
import os
from collections import namedtuple, OrderedDict
from os.path import splitext

import configargparse
import numpy as np
import pandas as pd
import spacy


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# --------------------------------------------------------------------------------
# OS functions

def _filterTxt(fullPath):
    (root, ext) = splitext(fullPath)
    return ext.lower() == '.txt'


def walkDirectory(path, filterFn):
    for currentDir, _, files in os.walk(path):
        # Get the absolute path of the currentDir parameter
        currentDir = os.path.abspath(currentDir)

        # Traverse through all files
        for fileName in files:
            fullPath = os.path.join(currentDir, fileName)

            if filterFn(fullPath):
                yield fullPath

# --------------------------------------------------------------------------------
# spaCy functions

def removeStopwords(doc, ignores):
    for token in doc:
        if (
            not token.is_stop and
            not token.is_punct and
            not token.is_digit and
            not token.is_space and
            not token.like_url and
            not token.text in ignores and
            token.pos_ != 'NUM'
        ):
            yield token

def bagOfWords(doc):
    # Load the SpaCy doc as a data frame (1 token per row)
    attrs = [
        [T.text, T.lemma_, T.pos_, 1]
        for T in removeStopwords(doc, [])
    ]
    return pd.DataFrame(attrs, columns=['word', 'lemma', 'pos', 'count'])


def buildCorpusDictionaries(df):
    # Create a pivot of lemma x part of speech
    reduced = df.groupby(['pos', 'lemma'])['count'].count().unstack(0)

    Z = df.groupby('lemma')['count'].sum()

    return {
        'total_docs': Z.max(),
        'all_lemmas': Z.to_dict(),
        'nouns': reduced.loc[reduced['NOUN'].notna(), 'NOUN'].to_dict(),
        'verbs': reduced.loc[reduced['VERB'].notna(), 'VERB'].to_dict()
    }


# --------------------------------------------------------------------------------
# Main

def build_arg_parser():
    p = configargparse.ArgParser(prog='build_corpus',
                                 description='builds a corpus from a directory')
    g = p.add_argument_group('Display')
    g.add('--handle-newline', default='keep',
          choices=['keep', 'ignore', 'space'],
          help='decide how newlines should be handled')
    g = p.add_argument_group('I/O')
    g.add('--directory', default=None,
          help='the directory to read')
    g.add('outfile',
          help='the output file')

    return p


def main():
    # Get the arguments from the command line
    p = build_arg_parser()
    options = p.parse_args()

    # Set the default output file name
    if options.directory is None:
        options.directory = os.getcwd()

    # Make the pipeline
    nlp = spacy.load('en_core_web_sm')

    # Initialize the outputs
    corpus_df = None

    # Get the list of files
    for fullPath in walkDirectory(options.directory, _filterTxt):
        print(f'Processing {fullPath}...')
        with io.open(fullPath,'r', encoding='utf-8') as f:
            raw = f.read()

        df = bagOfWords(nlp(raw))
        corpus_df = df if corpus_df is None else corpus_df.append(df)

    print(f'Making {options.outfile}...')
    result = buildCorpusDictionaries(corpus_df)
    with io.open(options.outfile,'w', encoding='utf-8') as f:
        json.dump(result, f, cls=NpEncoder)


if __name__ == '__main__':
    main()
