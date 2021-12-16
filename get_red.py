import cv2
import numpy as np

h = [(0, 10), (156, 180)]
s = [(43, 255)]
v = [(46, 255)]

red_ranges = [[(0, 43, 46), (20, 255, 255)], [(156, 43, 46), (180, 255, 255)]]


def get_red(image_bgr):
    images = np.zeros_like(image_bgr)[:, :, 0]
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    for r in red_ranges:
        images = images + cv2.inRange(image_hsv, r[0], r[1])
    return (images / 255).sum()


if __name__ == '__main__':
    import glob
    import sys
    import json

    data = [json.loads(x.strip()) for x in open(sys.argv[1])]

    with open(sys.argv[1] + 'red.txt', 'w') as f:
        for tmp in data:
            try:
                image = cv2.imread(tmp['image'])
                if get_red(image) > 0:
                    f.write(json.dumps(tmp, ensure_ascii=False) + '\n')
            except:
                print(tmp)
