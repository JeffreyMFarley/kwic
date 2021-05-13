import io
import json

import numpy as np
import pandas as pd


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


def buildCocaDicts(filename):
    def fix_word_class(x):
        c = x[:1]
        return c if c in ['n', 'v', 'j', 'r', 'm'] else 'x'

    table = pd.read_table(
        filename,
        usecols=['w1', 'L1', 'c1', 'coca', 'tcoca']
    )

    # Translate the old part of speech labels
    table['pos'] = table['c1'].apply(fix_word_class)

    # Create a pivot of lemma x part of speech
    reduced = table.groupby(['pos', 'L1'])['coca'].sum().unstack(0)

    # Create a summed series of lemma (part of speech agnostic)
    Z = table.groupby('L1')['coca'].sum()

    return {
        'total_docs': Z.max(),
        'all_lemmas': Z.to_dict(),
        'nouns': reduced.loc[reduced['n'].notna(), 'n'].to_dict(),
        'verbs': reduced.loc[reduced['v'].notna(), 'v'].to_dict(),
    }

# -----------------------------------------------------------------------------
# Main


print('Reading Corpus of Contemporary American English...')
coca = buildCocaDicts('../b240.txt')

print('Writing Results...')
with io.open('corpus.json', 'w', encoding='utf-8') as f:
    json.dump(coca, f, cls=NpEncoder)
