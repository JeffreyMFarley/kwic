import json
import io
import logging
import os
import sys
import traceback
import zipfile
from os.path import join, split, splitext
from urllib.parse import unquote_plus
from xml.etree.ElementTree import XML

import configargparse
import pdfplumber


BYTE_CR = '\n'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# --------------------------------------------------------------------------------
# Helpers

def buildWhitespaceReplace():
    table = {0xa0: ' ',    # non-breaking space

             0x2028: '\n',  # Line separator
             0x2029: '\n',  # Paragraph separator
             0x2060: ' ',  # Word-Joiner
             0x202f: ' ',  # Narrow no-break space
             0x205F: ' ',  # Medium Mathematical Space
             0x3000: ' ',  # Ideographic Space
             }
    table.update({c: ' ' for c in range(0x2000, 0x200b)}) # Unicode spaces
    table.update({c: None for c in range(0x200b, 0x200e)}) # Zero-width spaces

    return table


# --------------------------------------------------------------------------------
# Copy


def handle_copy(istream, outfile):
    pass

# --------------------------------------------------------------------------------
# Docx

def handle_docx(istream, outfile):
    """
    Take the path of a docx file as argument, return the text in unicode.
    (Inspired by python-docx <https://github.com/mikemaccana/python-docx>)
    """
    WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    PARA = WORD_NAMESPACE + 'p'
    TEXT = WORD_NAMESPACE + 't'

    document = zipfile.ZipFile(istream)
    xml_content = document.read('word/document.xml')
    document.close()
    tree = XML(xml_content)

    with io.open(outfile, 'w', encoding='utf-8') as ostream:
        for paragraph in tree.iter(PARA):
            texts = [node.text
                     for node in paragraph.iter(TEXT)
                     if node.text]
            if texts:
                ostream.write(''.join(texts))
                ostream.write(BYTE_CR)

    return f'docx converted to {outfile}'


# --------------------------------------------------------------------------------
# PDF

def handle_pdf(istream, outfile):
    previousPageHeader = ['', '', '', '', '']
    replacements = buildWhitespaceReplace()

    with pdfplumber.open(istream) as pdf:
        with io.open(outfile, 'w', encoding='utf-8') as ostream:
            for i, page in enumerate(pdf.pages):
                content = page.extract_text().translate(replacements)
                lines = content.split('\n')

                # - Logic for removing identical headers
                pageHeader = lines[0:4]
                thisPage = f'Page {i + 1}'
                same = set()

                for j, l in enumerate(pageHeader):
                    if l == previousPageHeader[j]:
                        same.add(j)
                    elif thisPage in l:
                        same.add(j)

                # Check last line
                if thisPage in lines[-1]:
                    same.add(len(lines) - 1)

                content = '\n'.join([
                    l.strip()
                    for j, l in enumerate(lines)
                    if j not in same
                ])

                if same:
                    ll = '\n'.join([lines[j] for j in list(same)])
                    logger.info(f'Page {i}: Removed Header(s)\n{ll}')

                previousPageHeader = pageHeader
                # /Logic

                # content = replace_characters(table, content, True)
                ostream.write(content)
                ostream.write(BYTE_CR)

    return f'PDF converted to {outfile}'


# --------------------------------------------------------------------------------
# Main

HANDLERS = {
    '.docx': handle_docx,
    '.pdf': handle_pdf,
    '.txt': handle_copy,
}

def build_arg_parser():
    p = configargparse.ArgParser(prog='to_text',
                                 description='extracts text from a PDF or DOC')
    g = p.add_argument_group('I/O')
    g.add('--infile', required=True,
          help='the input file')
    g.add('--outfile',
          help='the output file')

    return p


def main():
    response = ''

    # Get the arguments from the command line
    p = build_arg_parser()
    options = p.parse_args()

    # Get the extension
    (directory, filename) = split(options.infile)
    (root, ext) = splitext(filename)
    ext = ext.lower()

    # Early exit, unknown extension
    if ext not in HANDLERS:
        response = f"Unrecognized extension '{ext}'. No action taken."
        logger.warn(response)
        return response

    # Set the default output file name
    if options.outfile is None:
        root = root.replace(' ', '').replace('-', '').removesuffix('Interview')
        options.outfile = join(directory, root + '.txt')

    try:
        # Get the bytes from the new file
        # with io.open(options.infile, 'r', encoding='utf-8') as istream:
        #     # Convert the file
        response = HANDLERS[ext](options.infile, options.outfile)

    except Exception as ex:
        for fncall in traceback.format_exception(*sys.exc_info()):
            logger.debug(fncall)
        response = f'{ex}'
        logger.error(response)
        return response

    logger.info(response)
    print(response)

if __name__ == '__main__':
    main()
