import numpy as np
import json
import sys

if __name__ == '__main__':
    db = [json.loads(x.strip()) for x in open(sys.argv[1])]
    db_array = np.load(sys.argv[1] + '.npy')  # Nxd

    target = [json.loads(x.strip()) for x in open(sys.argv[2])]
    target_array = np.load(sys.argv[2] + '.npy')  # Mxd

    scores = np.dot(target_array, db_array.T)  # MxN
    target_ids, db_ids = np.where(scores > 0.9)

    similar = {}
    for target_id, db_id in zip(target_ids, db_ids):
        image = target[target_id]['image']
        if image not in similar:
            similar[image] = {'image': image, 'neighbor': [], 'neighbor_label': []}
        similar[image]['neighbor'].append(db[db_id]['image'])
        for k in db[db_id]:
            if k != 'image':
                if db[db_id][k] == 1:
                    similar[image]['neighbor_label'].append(k)

    with open(sys.argv[3], 'w') as f:
        for key in similar:
            similar[key]['neighbor_label'] = list(set(similar[key]['neighbor_label']))
            f.write(json.dumps(similar[key]) + '\n')
