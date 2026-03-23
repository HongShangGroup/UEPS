import numpy as np

def center_crop(data, shape):
    """
    center crop a tensor (or numpy ndarray) along the last two dimensions.

    Args:
        data: tensor/np.ndarray [..., H, W]
        shape: tuple (h, w)

    Returns:
        center cropped tensor.
    """
    H, W = data.shape[-2], data.shape[-1]
    h, w = shape
    assert 0 < h <= H
    assert 0 < w <= W

    w_from = (W - w) // 2
    h_from = (H - h) // 2
    w_to = w_from + w
    h_to = h_from + h
    data = data[..., h_from:h_to, w_from:w_to]
    return data

def img_rm_black_border(image, thre=1e-10):
    """cut image black border if exists

    Parameters:
        image: np array [H, W], non-negative
        thre: float, relative threshold for zero border

    Output:
        cropped_image: np array [h, w]
        border_idx: list of int, [top, bottom, left, right]
        border_exist: bool
    """
    assert len(image.shape) == 2
    assert np.min(image) >= 0
    assert np.max(image) > 0
    H, W = image.shape
    black_threshold = np.max(image) * thre

    # scan row
    row_means = np.mean(image, axis=1)
    non_black_rows = np.where(row_means > black_threshold)[0]
    top = non_black_rows[0]
    bottom = non_black_rows[-1]

    # scan column
    col_means = np.mean(image, axis=0)
    non_black_cols = np.where(col_means > black_threshold)[0]
    left = non_black_cols[0]
    right = non_black_cols[-1]

    # output
    cropped_image = image[top:bottom+1, left:right+1]
    border_idx = [top, bottom, left, right]

    if (top==0) and (bottom==(H-1)) and (left==0) and (right==(W-1)):
        border_exist = False
    else:
        border_exist = True

    return cropped_image, border_idx, border_exist

def img3_rm_black_border(A, B, C=None):
    """cut image black border if exists, cut B and C based on A

    Parameters:
        A: np array [1, H, W], non-negative
        B: np array [1, H, W]
        C: np array [1, H, W]

    Output:
        A: np array [1, h, w]
        B: np array [1, h, w]
        C: np array [1, h, w]
    """
    assert A.shape == B.shape
    assert len(A.shape) == 3
    assert A.shape[0] == 1

    if C is not None:
        assert A.shape == C.shape

    _, border_idx, border_exist = img_rm_black_border(A.squeeze(0))

    if border_exist:
        top, bottom, left, right = border_idx
        A = A[:, top:bottom+1, left:right+1]
        B = B[:, top:bottom+1, left:right+1]
        C = C[:, top:bottom+1, left:right+1] if C is not None else C

    return A, B, C
