from . import gpri_files as gpf

def gamma_dataset_generator(par_list, bin_list, **kwargs):
    #If something downstream fails, we want to inspect why
    value = None
    for par, bin in zip(par_list, bin_list):
        ds = gpf.gammaDataset(par, bin, **kwargs)
        if value is not None:
            print("type : {}".format(type(ds)))
            print(par, bin)
            pass
        value = yield ds