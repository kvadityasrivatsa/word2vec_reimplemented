import pickle as pkl

def load(fname):
    model = None
    with open(fname,'rb') as infile:
        model = pkl.load(infile)
    return model

def dump(obj,fname):
    with open(fname,'wb') as outfile:
        pkl.dump(obj,outfile)

