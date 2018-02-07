import requests
from lxml import html
import re
import argparse
import time
from random import randint

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-at', required=False, type=int, default=1)
    parser.add_argument('--output-file', required=False, default='corpus.txt')

    return parser.parse_args()

def parse_urls(tree):
    for url in tree.xpath('//td[@class="list-title"]/a/@href'):
        yield url


def get_next_page_url(tree):
    url_list = tree.xpath('//li[@class="pagination-next"]/a/@href')
    if url_list and len(url_list) == 0:
        return None

    return url_list[0]

def build_next_page_url(page_index):
    if page_index <= 1:
        return 'http://www.radiotraditional.ro/versuri-manele'
    if page_index <= 123:
        return 'http://www.radiotraditional.ro/versuri-manele/page-{}'.format(page_index)
    return None


def parse_page(tree):
    full_title = tree.xpath("//div[@class='item-page']/h1/a/text()")
    full_title = full_title[0] if full_title and len(full_title) > 0 else ''
    title = full_title.split('-')
    title = title[1] if len(title)>1 else title[0]
    title = title.strip()

    text = tree.xpath("//div[@class='item-page']")
    text = text[0]
    text = text.text_content()
    text = re.sub(r'^.+\d{4}\s\d{2}:\d{2}[^A-Z]', '', text)
    text = re.sub(r'\(.+;', '', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    return title, text

def sleep():
    num_seconds = randint(1, 15)
    time.sleep(num_seconds)

def run(args):
    root_url = 'http://www.radiotraditional.ro'
    page_index = args.start_at
    mode = 'at' if page_index > 1 else 'wt'
    page_url = build_next_page_url(page_index)
    with open(args.output_file, encoding='utf-8', mode=mode) as f:
        while(page_url):
            print("Parsing urls from page {}".format(page_url))
            root_page = requests.get(page_url)
            tree = html.fromstring(root_page.content)
            print("Done")

            sleep()
            for url in parse_urls(tree):
                text_url = root_url + url
                print("Parsing text from {}".format(text_url))
                sleep()
                text_page = requests.get(text_url)
                tp_tree = html.fromstring(text_page.content)
                title, text = parse_page(tp_tree)
                f.write(title)
                f.write('\n')
                f.write(text)
                f.write('\n')
            page_index += 1
            # page_url = root_url+ get_next_page_url(tree)
            page_url = build_next_page_url(page_index)


if __name__=='__main__':
    args = parse_arguments()
    run(args)
