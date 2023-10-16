import math

__all__ = ['pad_bbox', 'square_bbox']


def pad_bbox(xyxy, padding_ratio, H, W):
    x1, y1, x2, y2 = xyxy
    y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)

    if padding_ratio != 0:
        bbox_H = y2 - y1
        bbox_W = x2 - x1

        bbox_Hc = (y2 + y1) / 2
        bbox_Wc = (x2 + x1) / 2

        bbox_H2 = bbox_H * (1 + padding_ratio) / 2
        bbox_W2 = bbox_W * (1 + padding_ratio) / 2
        y1 = bbox_Hc - bbox_H2
        y2 = bbox_Hc + bbox_H2
        x1 = bbox_Wc - bbox_W2
        x2 = bbox_Wc + bbox_W2

        y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)
        y1 = max(0, y1)
        x1 = max(0, x1)

        x2 = min(W, x2)
        y2 = min(H, y2)
    return x1, y1, x2, y2


def square_bbox(xyxy, H, W):
    x1, y1, x2, y2 = xyxy
    h, w = y2 - y1, x2 - x1
    if h > w:
        c = int(math.ceil((x1 + x2) / 2))
        x1 = c - h // 2
        x2 = c + h - h // 2
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        elif x2 > W:
            x1 -= (x2 - W)
            x2 -= (x2 - W)
    elif w > h:
        c = int(math.ceil((y1 + y2) / 2))
        y1 = c - w // 2
        y2 = c + w - w // 2
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        elif y2 > H:
            y1 -= (y2 - H)
            y2 -= (y2 - H)
    return x1, y1, x2, y2
