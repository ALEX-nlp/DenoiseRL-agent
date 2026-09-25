import argparse
import sys
import json
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, '../')

from web_agent_site.utils import DEFAULT_FILE_PATH, DEFAULT_ATTR_PATH
from web_agent_site.engine.engine import load_products

parser = argparse.ArgumentParser(description="Convert WebShop products into search-index documents.")
parser.add_argument('--file-path', default=DEFAULT_FILE_PATH)
parser.add_argument('--attr-path', default=DEFAULT_ATTR_PATH)
parser.add_argument('--output-root', type=Path, default=Path('.'))
args = parser.parse_args()
all_products, *_ = load_products(filepath=args.file_path, attrpath=args.attr_path)


docs = []
for p in tqdm(all_products, total=len(all_products)):
    option_texts = []
    options = p.get('options', {})
    for option_name, option_contents in options.items():
        option_contents_text = ', '.join(option_contents)
        option_texts.append(f'{option_name}: {option_contents_text}')
    option_text = ', and '.join(option_texts)

    doc = dict()
    doc['id'] = p['asin']
    doc['contents'] = ' '.join([
        p['Title'],
        p['Description'],
        p['BulletPoints'][0],
        option_text,
    ]).lower()
    doc['product'] = p
    docs.append(doc)


for folder, limit in [('resources_100', 100), ('resources', None),
                      ('resources_1k', 1000), ('resources_100k', 100000)]:
    output = args.output_root / folder
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'documents.jsonl').open('w') as f:
        for doc in docs[:limit]:
            f.write(json.dumps(doc) + '\n')
