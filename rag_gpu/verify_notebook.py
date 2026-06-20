import json
from pathlib import Path

path = Path('conceptual-rag-gpu.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if any(marker in src for marker in [
        'def search_bg',
        'def search_sb',
        'load_docs',
        'bg_results = search_bg',
        'sb_results = search_sb',
        'def ask',
        'classify_query'
    ]):
        print(f'CELL {i}:')
        first_line = src.splitlines()[0] if src.splitlines() else ''
        print(first_line)
        print('---')

print('\nsearch_sb definitions =', sum('def search_sb' in ''.join(cell.get('source', [])) for cell in nb['cells']))
print('ask uses search_sb =', any('sb_results = search_sb' in ''.join(cell.get('source', [])) for cell in nb['cells']))
print('load docs markers =', any('load_docs' in ''.join(cell.get('source', [])) for cell in nb['cells']))
