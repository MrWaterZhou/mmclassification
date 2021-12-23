import cv2
import numpy as np


def random_ff_mask(height,width , max_angle=4, max_len=40, max_width=10, times=15):
    """Generate a random free form mask with configuration.
    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """

    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(times)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)
            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y
    return mask.astype(np.float32)

if __name__ == '__main__':
    image = random_ff_mask(640,512)
    cv2.imwrite('mask.jpg',image*255)
