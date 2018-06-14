import pandas as pd
import numpy as np

def load_data(path, fns, return_rn=False):
    res = ()
    for fn in fns:
        df = pd.read_csv(path+fn, index_col=[0])
        mat = df.as_matrix()
        print(df.shape)
        res += (mat,)
        if return_rn: res += (df.index,)
    if len(res)==1: res = np.array(res[0])
    return res

def load_sid2ca(path, fn='sid_ca.csv', rm_dup=True, dupabbr=['KIPAN', 'COADREAD', 'STES', 'GBMLGG'], verbose=True):
    df = pd.read_csv(path+fn, index_col=[0])
    if rm_dup: df = df.loc[~df['cancer'].isin(dupabbr)]
    sids = df.index
    cancers = df['cancer']
    if verbose: print('len: %d' % len(sids))
    return {s:c for s,c in zip(sids, cancers)}, set(list(cancers))

def load_gene2gid(path, fn='genes_gids.csv', verbose=True):
    df = pd.read_csv(path+fn)
    if verbose: print('len: %d' % len(df))
    return {g:i for g,i in zip(list(df['gene']), list(df['entrez']))}

def load_raw(path, fn='full.csv', rm_dup=True, rm_normal=True, dupabbr=['KIPAN', 'COADREAD', 'STES', 'GBMLGG'], verbose=False):
    df = pd.read_csv(path+fn, low_memory=False, index_col=[0])
    if rm_dup: df = df.loc[~df['cancer'].isin(dupabbr)]
    sids = list(df.index)
    df = df.assign(sid=sids)
    if rm_normal: df = df.loc[df.sid.str.split('-', expand=True).loc[:,3].str[0] != str(1)]
    df = df.iloc[:,:-2].as_matrix()
    if verbose: print('shape: %s' % str(df.shape))
    return df
