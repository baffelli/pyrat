import numpy as _np
import matplotlib.pyplot as plt

import matplotlib.patches as patch

def full_indices(index, overlap, bs):
    start = bs * index - overlap
    stop =  bs * index + bs + overlap
    return start, stop


class block:
    def __init__(self, data, location, block_shape, image_shape, overlap=[0, 0]):
        self.original_data = data
        self.overlap = overlap
        self.location = location
        self.block_shape = block_shape
        self.image_shape = image_shape
        #Start and stop indices



    def valid_indices(self):
        """
        Returns the valid indices of the block, that is the indices of the block that are
        inside the edges of the data array
        Returns
        -------

        """
        indices = ()
        clipped_indices = ()
        for index, overlap, bs, shp in zip(self.location, self.overlap, self.block_shape, self.original_data.shape):
            #clip everything outside of the array edges
            start = bs * index - overlap
            stop = bs * index + bs + overlap
            start_clip = _np.clip(start, 0 ,shp)
            stop_clip = _np.clip(stop, 0 ,shp)
            indices += (slice(start, stop),)
            clipped_indices += (slice(start_clip, stop_clip),)
        return indices, clipped_indices

    def overlap_pad(self):
        pads = ()
        indices, clipped_indices = self.valid_indices()
        for i, clip_i, ov, shp in zip(indices, clipped_indices, self.overlap, self.image_shape):
            start_pad = i.start % ov if i.start < 0 else 0
            stop_pad = i.stop % ov if i.stop > shp else 0
            pads += ((start_pad,stop_pad),)
        return pads

    def pads(self):
        """
        Computes the amount of padding necessary for a partial block,
        a block that is close to the lower right border of the image
        Returns
        -------

        """
        pads = ()
        #Compute the indices
        indices, clipped_indices = self.valid_indices()
        for i, clip_i in zip(indices, clipped_indices):
            start_pad = i.start - clip_i.start
            stop_pad = i.stop - clip_i.stop
            start_pad = -start_pad if start_pad < 0 else 0
            stop_pad = stop_pad if stop_pad > 0 else 0
            pads += ((start_pad,stop_pad),)
            #Compute the amount of trimming
        return pads
    #
    # def trimming(self):
    #     indices, clipped_indices = self.valid_indices()
    #     for i, clip_i, bs, ov in zip(indices, clipped_indices, self.block_shape, self.overlap):
    #         left_trim = ov
    #         right_trim =
    #







    @property
    def data(self):
        indices, clipped_indices = self.valid_indices()
        pads = self.pads()
        data = _np.pad(self.original_data[clipped_indices], pads + ((0,0),)*(self.original_data.ndim-2), mode='constant')
        return data

    def process(self, fun, trim=True):
        pad_cut = lambda x: -x  if x > 0 else None
        pads = self.pads()
        # Compute padding
        data_proc = fun(self)
        if trim and not _np.isscalar(data_proc):
            data_proc = data_proc[self.overlap[0]:pad_cut(self.overlap[0]),self.overlap[1]:pad_cut(self.overlap[1])]
        print(data_proc.shape)
        return data_proc


class block_array:
    def __init__(obj, A, block_size, overlap=[0, 0], pad_partial_blocks=False, trim_output=True, print_progress=True):
        # Create object
        obj.bs = block_size
        obj.overlap = overlap
        obj.nblocks = []
        # Compute shapes
        for current_shape, block_shape, overlap in zip(A.shape[0:2], block_size, overlap):
            obj.nblocks.append(int(_np.ceil(((current_shape) / (block_shape )))))
        #Compute padding for the partial blocks
        # if pad_partial_blocks:
        #     pads = _np.mod(-_np.array(A.shape[0:2]), block_size)
        #     pads = tuple([(0,int(p)) for p in pads]) + ((0,0),) * (A.ndim - 2)
        #     A = _np.pad(A,pads, mode='constant')
        # #Compute padding for the border
        # overlap_pad = tuple([(o,o) for o in overlap]) +  ((0, 0),) * (A.ndim - 2)
        # A = _np.pad(A, overlap_pad, mode='constant')
        # print(A.shape)
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
        # i_read, j_read = clip_indices((i, j), self.overlap, self.bs, self.A.shape)
        # Create block object
        current_block = block(self.A, (i, j), self.bs, self.A.shape, overlap=self.overlap,)
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
            # Apply function
            bl_out = bl.process(function, trim=self.trim_ouput)
            if self.print_progress:
                prog_str = "Processing block {i}, {j}, number {iter} of {maxiter} with shape {shp} and output shape {out_sh}".format(i=i, j=j, iter=self.current,
                                                                                          maxiter=self.maxiter, shp=bl.data.shape, out_sh = bl_out.shape)
                print(prog_str)

            #Concatenate them
            bl_arr[j][i] = bl_out
        arr = _np.hstack([_np.vstack(h_block) for h_block in bl_arr])
        return arr

    def __iter__(self):
        self.current = -1
        return self

    def __next__(self):
        print(self.current)
        if self.current >= self.maxiter - 1:
            # self.current = -1
            raise StopIteration
        else:
            self.current += 1
            c = self[self.current]
            return c
