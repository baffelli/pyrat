import numpy as _np


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
            obj.nblocks.append(current_shape // block_shape)
        obj.maxiter = _np.prod(obj.nblocks)

        obj.current = 0




    def block_index(self, idx):

        def valid_indices(index, overlap, bs, shp):
            start = _np.clip(bs * index - (overlap) ,0,shp)
            stop = _np.clip(bs * index + (overlap + bs),0,shp)
            return slice(start, stop)

        def pad_amount(index, overlap, bs, shp):
            pad_start = index * bs - (overlap)
            pad_stop = shp - (bs * index + (overlap+bs))
            ps = 0 if pad_start >=0 else -pad_start
            pe = None if pad_stop >=0 else bs-pad_stop
            sl = slice(ps,pe)
            return sl

        if idx < _np.prod(self.nblocks):
            i, j = _np.unravel_index(idx, self.nblocks)
            i_blk = valid_indices(i, self.overlap[0], self.bs[0], self.A.shape[0])
            j_blk = valid_indices(j, self.overlap[1], self.bs[1], self.A.shape[1])
            i_pad = pad_amount(i, self.overlap[0], self.bs[0], self.A.shape[0])
            j_pad = pad_amount(j, self.overlap[1], self.bs[1], self.A.shape[1])
            return i_blk, i_pad, j_blk, j_pad

    def __getitem__(self, idx):
        blk = _np.zeros((self.bs[0] + self.overlap[0]*2, self.bs[1] + self.overlap[1]*2) + self.A.shape[2::],
                        dtype=self.A.dtype)
        i, i_pad, j, j_pad = self.block_index(idx)
        A_cut = self.A[i,j]
        blk[i_pad, j_pad]  = A_cut
        return blk



    def __setitem__(self, sl, item):
        i, i_pad, j, j_pad = self.block_index(sl)
        blk = _np.zeros((self.bs[0] + self.overlap[0]*2, self.bs[1] + self.overlap[1]*2) + self.A.shape[2::],
                        dtype=self.A.dtype)
        blk[i_pad, j_pad] = item
        self.A[i,j] = blk

    def put_current(self, bl):
        self[self.current] = bl

    def process(self, function):
        for bl in self:
            print(self.current)
            self.put_current(function(bl))


    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.maxiter:
            self.current = 0
            raise StopIteration
        else:
            c = self[self.current]
            self.current += 1
            return c



