from skimage.draw import line

def func_drawline(mask_in, x1, y1, x2, y2, value=1):
    mask = mask_in.copy()
    rr, cc = line(int(y1), int(x1), int(y2), int(x2))
    mask[rr, cc] = value
    return mask
