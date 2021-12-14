import json
import os
import sys

if __name__ == '__main__':
    all_labels = sys.argv[3].split(',')
    with open(sys.argv[1]) as f:
        annotation = json.load(f)

    os.makedirs(os.path.dirname(sys.argv[2]),exist_ok=True)
    with open(sys.argv[2], 'w') as f:
        for data in annotation:
            if 'choice' in data:
                if isinstance(data['choice'], str):
                    labels = [data['choice']]
                else:
                    labels = data['choice']['choices']

                tmp = {'image': data['image']}
                for l in all_labels:
                    if l in labels:
                        tmp[l] = 1
                    else:
                        tmp[l] = 0
                f.write(json.dumps(tmp, ensure_ascii=False) + '\n')


