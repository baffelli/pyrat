import numpy as _np


def reading_indices(ij, overlaps, blocks, shape):
    sl = ()
    for index, overlap, bs, shp in zip(ij, overlaps, blocks, shape):
        start = _np.clip(bs * index - (overlap), 0, shp)
        stop = _np.clip(bs * index + (overlap + bs), 0, shp)
        sl += (slice(start, stop),)
    return sl

def writing_indices(ij, overlaps, blocks, shape):
    sl = ()
    for index, overlap, bs, shp in zip(ij, overlaps, blocks, shape):
        start = _np.clip(bs * index, 0, shp)
        stop = _np.clip(bs * index +  bs, 0, shp)
        sl += (slice(start, stop),)
    return sl

def padding_sizes(ij, overlaps, blocks, shape):
    pads = ()
    for index, overlap, bs, shp in zip(ij, overlaps, blocks, shape):
        pad_start = index * bs - (overlap)
        pad_stop = shp - (bs * index + (overlap + bs))
        ps = 0 if pad_start >= 0 else -pad_start
        pe = -pad_stop if pad_stop <= 0 else 0
        pads += ((ps, pe),)
    return pads
#
# def trimming_indices(ij, overlaps, blocks, shape):
#     trims = ()
#     pad = padding_sizes(ij, overlaps, blocks, shape)
#     for index, overlap, bs, shp in zip(ij, overlaps, blocks, shape):




class block_array:
    def __init__(obj, A, block_size, overlap=[0, 0]):

        # 2D shape of array
        shape_2d = A.shape[0:2]
        # Create object
        obj.bs = block_size
        obj.overlap = overlap
        obj.A = A
        obj.nblocks = []
        # Compute shapes
        for current_shape, block_shape, overlap_size in zip(A.shape, block_size, overlap):
            obj.nblocks.append(_np.floor(current_shape / block_shape))
        obj.maxiter = _np.prod(obj.nblocks)
        obj.current = -1

    def __getitem__(self, idx):
        # generate indices to read block
        i, j = _np.unravel_index(idx, self.nblocks)
        i_read, j_read = reading_indices((i, j), self.overlap, self.bs, self.A.shape)
        # read it
        A_cut = self.A[i_read, j_read]
        # Compute padding size
        pads = padding_sizes((i, j), self.overlap, self.bs, self.A.shape)
        # pad
        A_cut = _np.pad(A_cut, pads, mode='constant')
        return A_cut

    def __setitem__(self, sl, item):
        #get location of block
        i, j = _np.unravel_index(sl, self.nblocks)
        i_read, j_read = reading_indices((i, j), self.overlap, self.bs, self.A.shape)
        i_write, j_write = writing_indices((i, j), self.overlap, self.bs, self.A.shape)
        pad_i, pad_j = padding_sizes((i, j), self.overlap, self.bs, self.A.shape)
        #Cut out the padded part
        if _np.isscalar(item):
            self.A[i_write, j_write] = item
        else:
            print(pad_i, pad_j)
            print(i_write, j_write)
            end_cut = lambda x: -1 if x == 0 else x
            item_cut = item[(self.overlap[0]):-(self.overlap[0]),(self.overlap[1]):-(self.overlap[1])]
            self.A[i_write, j_write] = item_cut

    def put_current(self, bl):
        self[self.current] = bl

    def __copy__(self):
        new = type(self)(self.A * 1, self.bs, overlap=self.overlap)
        new.__dict__.update(self.__dict__)
        #Copy array
        new.A = self.A * 1
        return new

    def process(self, function):
        A_copy = self.__copy__()
        for bl in self:
            A_copy[self.current] = function(bl)
        return A_copy

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.maxiter - 1:
            self.current = 0
            raise StopIteration
        else:
            self.current += 1
            c = self[self.current]
            return c
