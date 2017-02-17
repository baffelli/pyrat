import numpy as _np


def reading_indices(ij, overlaps, blocks, shape):
    sl = ()
    for index, overlap, bs, shp in zip(ij, overlaps, blocks, shape):
        start = _np.clip(bs * index - (overlap), 0, shp)
        stop = _np.clip(bs * index + (overlap + bs), 0, shp)
        sl += (slice(start, stop),)
    return sl


class block:
    def __init__(self, data, location, block_shape, image_shape, overlap=[0, 0], pad_partial_blocks=False, ):
        self.original_data = data
        self.overlap = overlap
        self.location = location
        self.block_shape = block_shape
        self.image_shape = image_shape
        # Shape with padding
        self.pad_partial_blocks = pad_partial_blocks

    def compute_padding(self):
        overlap_pads = ()
        partial_pads = ()
        trimming = ()
        ps = lambda p: 0 if p > 0 else -p
        pe = lambda p: -p if p < 0 else 0
        for index, overlap, bs, shp in zip(self.location, self.overlap, self.block_shape, self.image_shape):
            # This is where the block starts
            block_start = index * bs
            # Pad for the overlap
            overlap_pad_start = ps(block_start - (overlap))
            overlap_pad_stop = pe(shp - (block_start + bs))
            # Pad for partial block
            pad_start = 0
            pad_stop = pe(shp - (block_start + (bs * self.pad_partial_blocks + overlap)))
            # Add
            overlap_pads += ((overlap_pad_start, overlap_pad_stop),)
            partial_pads += ((pad_start, pad_stop),)
            # Compute the additional amount of trimming
            trimming += ((abs(pad_start - overlap_pad_start), abs(pad_stop - overlap_pad_stop)),)
        # if not self.pad_partial_blocks:
        #     partial_pads = ((0, 0), (0, 0))

        return overlap_pads, partial_pads, trimming

    @property
    def data(self):
        # Padding for partial blocks
        overlap_paddnig, partial_padding, trimming = self.compute_padding()
        # Extra padding for the additional dimensions
        extra_pad = ((0, 0),) * (self.original_data.ndim - 2)
        block_pad = _np.pad(_np.pad(self.original_data, overlap_paddnig + extra_pad, mode='constant'), partial_padding+extra_pad, mode='constant')
        # If the data is not of the size of the block, pad it
        return block_pad

    def process(self, fun, trim=True):
        pad_cut = lambda x: -x if x > 0 else None
        # Compute padding
        overlap_pad, partial_pad, trimming = self.compute_padding()
        data_proc = fun(self.data)
        if trim and not _np.isscalar(data_proc):
            data_proc = data_proc[self.overlap[0]:pad_cut(self.overlap[0] + trimming[0][1]),
                        self.overlap[1]:pad_cut(self.overlap[1] + trimming[1][1])]
        # f, (a1,a2) = plt.subplots(2,1)
        # a1.imshow(data_proc)
        # a2.imshow(self.data)
        # plt.show()
        return data_proc


class block_array:
    def __init__(obj, A, block_size, overlap=[0, 0], pad_partial_blocks=False, trim_output=True, print_progress=True):
        # Create object
        obj.bs = block_size
        obj.overlap = overlap
        obj.A = A
        obj.nblocks = []
        # Compute shapes
        for current_shape, block_shape, overlap_size in zip(A.shape[0:2], block_size, overlap):
            obj.nblocks.append(int(_np.ceil(current_shape / block_shape)))
        obj.maxiter = _np.prod(obj.nblocks)
        obj.current = -1
        obj.pad_partial_blocks = pad_partial_blocks
        obj.trim_ouput = trim_output
        obj.print_progress = print_progress

    def __getitem__(self, idx):
        # generate indices to read block
        i, j = _np.unravel_index(idx, self.nblocks)
        # generate block structure
        i_read, j_read = reading_indices((i, j), self.overlap, self.bs, self.A.shape)
        # Create block object
        current_block = block(self.A[i_read, j_read], (i, j), self.bs, self.A.shape, overlap=self.overlap,
                              pad_partial_blocks=self.pad_partial_blocks)
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
                prog_str = "Processing block {i}, {j}, number {iter} of {maxiter}".format(i=i, j=j, iter=self.current,
                                                                                          maxiter=self.maxiter)
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
