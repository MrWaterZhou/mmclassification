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
            similar[image] = {'image': image, 'neighbor': []}
        similar[image]['neighbor'].append(db[db_id]['image'])

    with open(sys.argv[3], 'w') as f:
        for key in similar:
            f.write(json.dumps(similar[key]) + '\n')


