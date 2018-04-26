from bs4 import BeautifulSoup
import os
import re
import logging
import argparse

LOG = logging.getLogger(__name__)


def parse_files(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            LOG.info("Parsing file {}".format(file_path))
            with open(file_path) as f:
                soup = BeautifulSoup(f, 'html.parser')
                text = soup.find('div', class_='component-content').get_text()
                text = re.sub(r"<[^>]+>", '', text)    # replace imbricated html tags
                text = re.sub(r"\([^)]+\)", '', text)  # replace scripts
                text = re.sub(r"\.push;", '', text)     # replace remainder scripts
                text = re.sub(r"Versuri[^-]+-\s", '', text)  # replace the author
                text = re.sub(r"Detalii.+\d\d:\d\d\s+", '\n', text)
                yield text


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', help='The path to the input directory containing files to parse.', required=True)
    parser.add_argument('--output-file', help='The path to the output file.', required=True)
    args = parser.parse_args()
    return args


def run(input_dir, output_file):
    with open(output_file, 'wt') as f:
        for _, text in enumerate(parse_files(input_dir)):
            f.write(text)
    LOG.info("That's all folks!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s:%(name)s %(funcName)s: %(message)s')
    args = parse_arguments()
    run(args.input_dir, args.output_file)
