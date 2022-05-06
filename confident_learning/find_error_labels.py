import numpy as np
from cleanlab.filter import find_label_issues
import sys
import json

if __name__ == '__main__':
    labels_to_check = ['性感_胸部', '轻度性感_胸部']
    data = [json.loads(x.strip()) for x in open(sys.argv[1])]
    labels = []
    preds = []

    for x in data:
        label = []
        pred = []
        normal_prob = 1
        for i, l in enumerate(labels_to_check):
            if x['labels'][l] == 1:
                label.append(i)
            pred.append(x['preds'][l])
            normal_prob = normal_prob * (1 - x['preds'][l])

        if len(label) == 0:
            label.append(len(labels_to_check))
        pred.append(normal_prob)
        labels.append(label)
        preds.append(pred)
    preds = np.array(preds)

    err_boolean = find_label_issues(labels=labels, pred_probs=preds, multi_label=True)

    with open(sys.argv[2], 'w') as f:
        for x, y in zip(data, err_boolean):
            if y:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')
