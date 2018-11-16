from __future__ import print_function
from argparse import ArgumentParser

from fastai.learner import *
from fastai.column_data import *

import numpy as np
import pandas as pd

def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--data', type=str, nargs=None, dest='in_path', help='input file path', required=True)

    parser.add_argument('--out-prefix', type=str, nargs=None, dest='model', help='output prefix', required=True)

    parser.add_argument('--out-dir', type=str, nargs=None, dest='out_dir', help='output directory', required=True)

    parser.add_argument('--num-dim', type=int, nargs=None, dest='num_dim', help='number of dimension of resulting embedding', required=False, default=50)

    parser.add_argument('--bs', type=int, nargs=None, dest='bs', help='batch size', required=False, default=64)

    parser.add_argument('--num-epoch', type=int, nargs=None, dest='num_eps', help='number of epoch(s)', required=False, default=3)

    parser.add_argument('--learning-rate', type=float, nargs=None, dest='lr', help='learning rate', required=False, default=1e-5)

    return parser

def main():
    parser = build_parser()
    opts = parser.parse_args()

    if torch.cuda.is_available() and torch.backends.cudnn.enabled:
        torch.cuda.set_device(0)
    else:
        print('CUDA or CUDNN not available.')
        return

    in_path = opts.in_path
    n_factors = opts.num_dim
    bs = opts.bs
    num_eps = opts.num_eps
    lr = opts.lr
    out_dir = opts.out_dir
    prefix = opts.model
    outpath = out_dir+'/'+prefix+'_'

    ### data preparation
    df = pd.read_csv(in_path, sep=',', low_memory=False, index_col=[0], error_bad_lines=False)
    sids = list(df.index)
    df = df.assign(id=sids)
    df = df.reset_index(drop=True)
    mdf = pd.melt(df, id_vars=['id'], var_name='gene', value_name='log2exp')

    ### training
    val_idxs = get_cv_idxs(len(mdf))
    cd = CollabFilterDataset.from_data_frame(path, mdf, 'id', 'gene', 'log2exp')
    learn = cd.get_learner(n_factors, val_idxs, bs, opt_fn=optim.Adam)
    learn.fit(lr, num_eps)
    learn.save(outpath+'model')

    ### plot jointplot
    preds = learn.predict()
    y=learn.data.val_y
    jp = sns.jointplot(preds, y, kind='hex', stat_func=None)
    jp.set_axis_labels('ground truth log2(exp)', 'predicted log2(exp)')
    jp.savefig(outpath+'trn_metric_jointplot.png')

    ### output embedding
    genes = list(df.columns[:-2])
    sids = list(df['id'])
    geneidx = np.array([cd.item2idx[g] for g in genes])

    m=learn.model
    m.cuda()

    ### output gene embedding matrix and bias
    gene_emb = to_np(m.i(V(geneidx)))
    gene_emb_df = pd.DataFrame(gene_emb, index=genes)
    gene_emb_df.to_csv(outpath+'gemb.csv', sep=',')
    gene_emb_bias = to_np(m.ib(V(geneidx)))
    gene_emb_bias_df = pd.DataFrame(gene_emb_bias, index=genes)
    gene_emb_bias_df.to_csv(outpath+'gemb_bias.csv')

    ### output sample embedding matrix and bias
    sampleidx = np.array([cd.user2idx[sid] for sid in sids])
    samp_emb = to_np(m.u(V(sampleidx)))
    samp_emb_df = pd.DataFrame(samp_emb, index=sids)
    samp_emb_df.to_csv(outpath+'semb.csv', sep=',')
    samp_emb_bias = to_np(m.ub(V(sampleidx)))
    samp_emb_bias_df = pd.DataFrame(samp_emb_bias, index=sids)
    samp_emb_bias_df.to_csv(outpath+'semb_bias.csv')

if __name__ == '__main__':
    main()
