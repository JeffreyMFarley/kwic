# kwic
Display the key words in context

## Installation

```
git clone https://github.com/JeffreyMFarley/kwic.git
cd kwic
<set up your virtual python environment>
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Pre-requistes

You will need a `corpus.json` that contains the document frequency for each lemma in English.
See [this commit](https://github.com/JeffreyMFarley/kwic/commit/78264556ad39623aaaf36b7127f65ef24ecc4bc6) for more details

## Running the program

```
$ python kwic.py --help                                                 
usage: kwic [-h] [--top TOP] [--min-occ MIN_OCC]
            [--handle-newline {keep,ignore,space}]
            document

shows key words in context

optional arguments:
  -h, --help            show this help message and exit

Analysis:
  --top TOP             how many high scoring nouns and verbs are keywords
  --min-occ MIN_OCC     number of key words in an important sentence

Display:
  --handle-newline {keep,ignore,space}
                        decide how newlines should be handled

I/O:
  document              the file to analyze
```

## Example

```
$ python kwic.py --handle-newline space sample_texts/moby-dick.txt                     
```
