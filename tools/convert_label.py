import sys
import json

if __name__ == '__main__':
    patch = json.load(open(sys.argv[1]))
    labels = ["性感_胸部", "色情_女胸", "性感_男性胸部"]
    # labels = ["色情_裸露下体"]
    patch_json = {}
    for x in patch:
        choices = {c: 0 for c in labels}
        if 'choice' in x:
            if isinstance(x['choice'], str):
                choices[x['choice']] = 1
            else:
                for c in x['choice']['choices']:
                    choices[c] = 1
        patch_json[x['image']] = choices

    data = open(sys.argv[2]).readlines()

    data = [json.loads(x.strip()) for x in data]
    save = []
    for x in data:
        if x['image'] in patch_json:
            x.update(patch_json[x['image']])
            save.append(x)

    with open(sys.argv[2] + '.patched', 'w') as f:
        for x in save:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
