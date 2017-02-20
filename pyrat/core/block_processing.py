import numpy as _np
import matplotlib.pyplot as plt



def full_indices(index, overlap, bs):
    start = bs * index
    stop =  bs * index + bs + overlap * 2
    return start, stop

def clip_indices(ij, overlaps, blocks, shape):
    sl = ()
    for index, overlap, bs, shp in zip(ij, overlaps, blocks, shape):
        start, stop, = full_indices(index, overlap, bs)
        sl += (slice(start, stop),)
    return sl


class block:
    def __init__(self, data, location, block_shape, image_shape, overlap=[0, 0]):
        self.original_data = data
        self.overlap = overlap
        self.location = location
        self.block_shape = block_shape
        self.image_shape = image_shape



    @property
    def data(self):
        return self.original_data

    def process(self, fun, trim=True):
        pad_cut = lambda x: -x if x > 0 else None
        # Compute padding
        data_proc = fun(self.data)
        if trim and not _np.isscalar(data_proc):
            data_proc = data_proc[self.overlap[0]:pad_cut(self.overlap[0]),
                        self.overlap[1]:pad_cut(self.overlap[1])]
        return data_proc


class block_array:
    def __init__(obj, A, block_size, overlap=[0, 0], pad_partial_blocks=False, trim_output=True, print_progress=True):
        # Create object
        obj.bs = block_size
        obj.overlap = overlap
        obj.nblocks = []
        # Compute shapes
        for current_shape, block_shape in zip(A.shape[0:2], block_size):
            obj.nblocks.append(int(_np.ceil(current_shape / block_shape)))
        #Compute padding for the partial blocks
        if pad_partial_blocks:
            pads = _np.mod(-_np.array(A.shape[0:2]), block_size)
            pads = tuple([(0,int(p)) for p in pads]) + ((0,0),) * (A.ndim - 2)
            A = _np.pad(A,pads, mode='constant')
        #Compute padding for the border
        overlap_pad = tuple([(o,o) for o in overlap]) +  ((0, 0),) * (A.ndim - 2)
        A = _np.pad(A, overlap_pad, mode='constant')
        print(A.shape)
        obj.A = A
        obj.maxiter = _np.prod(obj.nblocks)
        obj.current = -1
        obj.pad_partial_blocks = pad_partial_blocks
        obj.trim_ouput = trim_output
        obj.print_progress = print_progress

    def __getitem__(self, idx):
        # generate indices to read block
        i, j = _np.unravel_index(idx, self.nblocks)
        # generate block structure
        i_read, j_read = clip_indices((i, j), self.overlap, self.bs, self.A.shape)
        # Create block object
        current_block = block(self.A[i_read, j_read], (i, j), self.bs, self.A.shape, overlap=self.overlap,)
        return current_block

    def __copy__(self):
        new = type(self)(self.A * 1, self.bs, overlap=self.overlap)
        new.__dict__.update(self.__dict__)
        # Copy array
        new.A = self.A * 1
        return new

    def process(self, function):
        bl_arr = [[0 for i in range(self.nblocks[0])] for j in range(self.nblocks[1])]
        for bl in self:
            i, j = _np.unravel_index(self.current, self.nblocks)
            if self.print_progress:
                prog_str = "Processing block {i}, {j}, number {iter} of {maxiter} with shape {shp}".format(i=i, j=j, iter=self.current,
                                                                                          maxiter=self.maxiter, shp=bl.data.shape)
                print(prog_str)
            # Apply function
            bl_out = bl.process(function, trim=self.trim_ouput)
            # Trim if necessary
            bl_arr[j][i] = bl_out
        arr = _np.hstack([_np.vstack(h_block) for h_block in bl_arr])
        return arr

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
