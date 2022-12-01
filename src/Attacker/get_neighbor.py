import numpy as np
from skimage.segmentation import slic

from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class Neighbor(object):
    """
    compute 
    """

    def __init__(self, *args, **kwargs):
        self.current_index = np.zeros(1,)
        
    def get(self, *args, **kwargs):
        if True: # この条件分岐をクラス内変数でやれば拡張簡単だと思います。
            return self._simple(*args, **kwargs)
        else:
            raise NotImplementedError()
    
    def _simple(self, targets, idx, superpixel, is_upper, *args, **kwargs):
        c, label = targets[idx][int(self.current_index[idx])]
        # searched[idx].append((c, label))
        # targets[idx] = np.delete(targets[idx], 0, axis=0)
        is_upper[idx, c, superpixel[idx] == label] = ~is_upper[
            idx, c, superpixel[idx] == label
        ]
        self.current_index[idx] += 1
        return is_upper
    
    def reset(self, N=1):
        self.current_index = np.zeros(N, )


class SuperPixel(object):
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def get(self, *args, **kwargs):
        if True:
            return self.cal_superpixel(*args, **kwargs)
        elif False:
            return
        else:
            raise NotImplementedError()
    
    def cal_superpixel(self, x, idx, total, *args, **kwargs):
        pbar.debug(idx + 1, total, "cal_superpixel")
        superpixel_storage = []
        for n_segments in config.segments:
            img = (x.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            superpixel = slic(img, n_segments=n_segments)
            superpixel_storage.append(superpixel)
        return superpixel_storage
    
    def cal_squares(self, x, idx, total, *args, **kwargs):
        """WIP
        """
        pbar.debug(idx + 1, total, "cal_square")
        superpixel_storage = []
        for n_segments in config.segments:
            img = (x.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            # superpixel = # 正方形を作る
            superpixel = np.zeros_like(img)
            superpixel_storage.append(superpixel)
        return superpixel_storage