import json
import os

path = r'd:\Lab\BDA\Project\notebooks\02_text_classification_models.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The cell we want to modify is the 3rd cell (index 2)
# Verified from view_file output:
# 32:       "cell_type": "code",
# 33:       "execution_count": 9,
# 57:       "source": [

cell = nb['cells'][2]
source = cell['source']

new_source = []
for line in source:
    if 'GITHUB_TOKEN = userdata.get' in line:
        new_source.append(line)
        new_source.append('    # !!! UPDATE WITH YOUR OWN GITHUB DETAILS !!!\n')
        continue
    if '!git config --global user.email' in line:
        new_source.append(line)
        continue
    if '!git config --global user.name' in line:
        new_source.append(line)
        new_source.append('    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
        continue
    new_source.append(line)

cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
