import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.neighbors import DistanceMetric
import networkx as nx
from itertools import combinations

def undo_PCA(x, pca, pca_comp):
    mu = np.mean(x, axis=0)
    xhat = np.dot(pca.transform(x), pca_comp)
    xhat += mu
    print(xhat.shape)
    return xhat.T

def emb2exp(semb, gemb, semb_bias, gemb_bias):
    x = np.dot(semb, gemb.T)
    x += semb_bias
    x += gemb_bias.T
    print(x.shape)
    return x

def plot_samp_3dPCA(samp_pca, cats, cat2col,
                    subset_idxs=[], showcat=False, showlegend=True,
                    alpha=0.1, s=25, fs=(20,20)):
    if len(subset_idxs)==0:
        X = samp_pca[0]
        Y = samp_pca[1]
        Z = samp_pca[2]
    else:
        X = samp_pca[0][subset_idxs]
        Y = samp_pca[1][subset_idxs]
        Z = samp_pca[2][subset_idxs]
    colors = [cat2col[c] for c in cats]

    fig = plt.figure(figsize=fs)
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, Z, c=colors, s=s, alpha=alpha)
    if showcat:
        for x, y, z, c in zip(X, Y, Z, cats): ax.text(x, y, z, c, fontsize=8)
    if showlegend:
        proxies = []
        for c in cat2col: proxies.append(plt.Rectangle((0, 0), 1, 1, fc=cat2col[c]))
        ax.legend(proxies, list(set(list(cat2col))), numpoints = 1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

def plot_gene_3dPCA(gene_pca, genelist,
                    hl_idxs=[], hl_cols=['r', 'g', 'b'],
                    showhlgene=True, showbg=True,
                    bgcol=(0.5,0.5,0.5), bgs=30, bgalpha=0.1,
                    hlfs=10, hls=30, hlalpha=0.8, fs=(20,20)):
    X = gene_pca[0]
    Y = gene_pca[1]
    Z = gene_pca[2]

    fig = plt.figure(figsize=fs)
    ax = fig.gca(projection='3d')
    if showbg: ax.scatter(X, Y, Z, c=(0.5,0.5,0.5), s=bgs, alpha=bgalpha)
    for i, hl in enumerate(hl_idxs):
        for idx in hl:
            if showhlgene: ax.text(X[idx],Y[idx],Z[idx],genelist[idx], color=hl_cols[i], fontsize=hlfs)
            ax.scatter(X[idx],Y[idx],Z[idx], c=hl_cols[i], s=hls, alpha=hlalpha)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

def find_centroid(emb, sid2ca, sids, cancer):
    idxs = [i for i,s in enumerate(sids) if sid2ca[s]==cancer]
    arr = np.array([emb[i] for i in idxs])
    return np.mean(arr, axis=0)

def print_gdist(g1, g2, dist, gene2idx):
    print(dist[gene2idx[g1]][gene2idx[g2]])

def get_emb_dist(emb, return_pd=True, index=[], distmetric='euclidean', masked=False, maskval=10):
    dist = DistanceMetric.get_metric(distmetric)
    emb_dist = np.absolute(dist.pairwise(emb))
    if masked:
        utri = np.triu_indices(len(emb_dist))
        emb_dist_masked = emb_dist
        emb_dist_masked[utri] = maskval
        res = emb_dist_masked
    else:
        res = emb_dist
    if return_pd: res = pd.DataFrame(res, index=index, columns=index)
    print('shape: %s; mean: %.3f; std: %.3f' % (str(emb_dist.shape), emb_dist.mean(), emb_dist.std()))
    mean = np.mean(emb_dist, axis=0)
    std = np.std(emb_dist, axis=0)
    return res, mean, std

def n_closest_nbs(dist, gene, n=10):
    n += 1
    arr = np.array(dist[gene])
    nb_idx = np.argpartition(arr, n)[:n]
    tdf = pd.DataFrame(dist[gene][nb_idx]).T.assign(parent=gene)
    mdf = pd.melt(tdf, id_vars='parent', var_name='child', value_name='l2dist')
    mdf = mdf[mdf.child != gene]
    return mdf

def pull_close_nbs(dist, gene, th, distmetric='l2dist'):
    arr = np.array(dist[gene])
    nb_idx = np.where(arr < th)[0]
    tdf = pd.DataFrame(dist[gene][nb_idx]).T.assign(parent=gene)
    mdf = pd.melt(tdf, id_vars='parent', var_name='child', value_name=distmetric)
    mdf = mdf[mdf.child != gene]
    return mdf

def get_close_nbs_mdf(dist, genes, th, verbose=True):
    dist_nbs_mdfs = [pull_close_nbs(dist, g, th=th) for g in genes]
    dist_nbs_mdf = pd.concat(dist_nbs_mdfs)
    if verbose: print('len: %d' % len(dist_nbs_mdf))
    return dist_nbs_mdf

def get_connectivity_score(mdf):
    arr = list(mdf['parent']) + list(mdf['child'])
    genes = list(set(arr))
    print('counting gene connectivity...')
    counts = [arr.count(g) for g in genes]
    total_count = np.sum(counts)
    print('calculating score...')
    percent = [c/total_count for c in counts]
    pds = pd.Series(percent, index=genes)
    print('# of genes: %d' % len(genes))
    return pds, genes

def get_cs_df(mdf1, mdf2, colname=['cancer', 'normal'], get_diff=False, show_zscore=False, show_outliers=False, z_cutoff=3):
    print('getting score of set 1...')
    cs1, g1 = get_connectivity_score(mdf1)
    print('getting score of set 2...')
    cs2, g2 = get_connectivity_score(mdf2)
    print('# of common genes: %d' % len(list(set(g1) & set(g2))))
    cs1_df = cs1.to_frame(); cs1_df.columns = [colname[0]]
    cs2_df = cs2.to_frame(); cs2_df.columns = [colname[1]]
    cs_df = pd.concat([cs1_df, cs2_df], axis=1)
    cs_df = cs_df.fillna(0)
    if get_diff:
        cs_df = cs_df.assign(diff=cs_df.iloc[:,0]-cs_df.iloc[:,1])
        if show_zscore:
            diff = list(cs_df['diff'])
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            zscore_diff = [(x - mean_diff)/std_diff for x in diff]
            cs_df = cs_df.assign(zscore=zscore_diff)
            if show_outliers:
                outliers = np.where(np.absolute(zscore_diff) > z_cutoff, np.where(np.array(zscore_diff) > z_cutoff, colname[0], colname[1]), np.nan)
                cs_df = cs_df.assign(outliers=outliers)
    return cs_df

def pull_cluster_genes(x, y, xr, yr):
    ### x,y is pd series
    cg = list(set(x[x.between(xr[0], xr[1])].index) & set(y[y.between(yr[0], yr[1])].index))
    print('# of genes: %d' % len(cg))
    print(len(cg))
    return cg

def export_gid(path, fn, genes, gene2gid):
    print('saving to %s' % path+fn)
    with open(path+fn, 'w') as f:
        for g in genes:
            f.write("%s\n" % gene2gid[g])
    print('saved')

def pull_top_genes(emb, dim, genes, n=100, tail=False):
    arr = np.array(emb.T[dim])
    if tail:
        idx = np.argpartition(arr, n)[:n]
    else:
        idx = np.argpartition(arr, -n)[-n:]
    return genes[idx]

def subset_semb(emb, cancer, sids, sid2ca, drop_lowvar=True):
    idx = [i for i,s in enumerate(sids) if sid2ca[s]==cancer]
    tmp = pd.DataFrame(emb[idx], index=sids[idx])
    if drop_lowvar:
        std = np.std(tmp, axis=0)
        mean_std = std.mean()
        colidx = [i for i,std in enumerate(std) if std>mean_std]
        res = tmp.iloc[:,colidx]
    else:
        res = tmp
    return res

def show_genes_dim(emb, genes, gene2idx, fs=(12,5), cmap='RdBu_r'):
    idx = [gene2idx[g] for g in genes]
    tmp = pd.DataFrame(emb[idx], index=genes)
    plt.figure(figsize=fs)
    sns.heatmap(tmp, xticklabels=True, yticklabels=True, cmap=cmap, center=0)
    plt.show()

def get_sig_df(df, th_size=10, th_fdr=1.3, th_pval=2, out_fil=False, out_bp=False):
    tFDR = np.log10(df['Benjamini and Hochberg (FDR)'])*-1
    tpval = np.log10(df['P-value'])*-1
    tFDR[tFDR == np.inf] = 20
    tpval[tpval == np.inf] = 20
    mdf = pd.DataFrame({'GO':df['Process'],
                 'name':df['name'],
                 'size':df['num_of_Genes'],
                 'pval':tpval,
                 'FDR':tFDR,
                  'genes':df['Genes']})
    fil_df = mdf.loc[mdf['size']>th_size]
    fil_df = fil_df.sort_values('FDR', ascending=False)
    fil_df = fil_df.reset_index(drop=True)
    sig_df = fil_df.loc[(fil_df['FDR']>th_fdr) & (fil_df['pval']>th_pval)]
    sig_df = sig_df.sort_values('FDR', ascending=False)
    sig_df = sig_df.reset_index(drop=True)
    if out_fil:
        res = (sig_df, fil_df)
    else:
        res = sig_df
    if out_bp:
        bp_df = fil_df.sort_values('size', ascending=False)
        bp_df = bp_df.reset_index(drop=True)
        res += (bp_df, )
    return res

def plot_go_bubble(fil_df, sig_df, print_labs=True, th_fdr=1.3, gs_prox=[10,30], fs=(8,10), style='seaborn', szf=10, bgalpha=0.3, sigalpha=0.5, cmap='YlOrRd', lnalpha=0.5, siglabfs=10, labfs=15):
    x = range(len(fil_df))
    y = fil_df['FDR']
    x_sig = range(len(sig_df))
    y_sig = sig_df['FDR']
    fig = plt.figure(figsize=fs)
    plt.style.use(style)
    plt.scatter(x, y, s=fil_df['size']*szf, c='grey', alpha=bgalpha)
    plt.scatter(x_sig, y_sig, s=sig_df['size']*szf, c=y_sig, cmap=cmap, alpha=sigalpha)
    if print_labs:
        for i in x_sig:
                plt.text(i, y_sig[i], sig_df['GO'][i], fontsize=siglabfs)
    plt.axhline(y=th_fdr, linewidth=1, linestyle='--', color='r', alpha=lnalpha)
    proxies = [plt.scatter([], [], s=sz*szf, c='k', alpha=alpha) for sz in gs_prox]
    plabs = [str(sz) for sz in gs_prox]
    plt.legend(proxies, plabs, title='Gene Set Size', loc=4)
    plt.xlabel('rank', fontsize=labfs)
    plt.ylabel('-log10(FDR)', fontsize=labfs)
    plt.show()

def plot_go_bp(bp_df, fs=(8,8), style='seaborn', labfs=15):
    x = range(len(bp_df))
    y = list(bp_df['size'])
    xlabs = bp_df['GO']
    fig = plt.figure(figsize=fs)
    plt.style.use(style)
    plt.bar(x, y)
    plt.xlabel('GO term', fontsize=labfs)
    plt.ylabel('number of genes', fontsize=labfs)
    plt.show()

def plot_go_bubble_compare(sig_dfs, xlabs, szf=10, gs_prox=[10,30],
fs=(7,10), style='seaborn', cmaps=['Blues', 'Blues', 'Wistia', 'Wistia', 'Greens', 'Greens', 'Reds', 'Reds'], vmin=-5, vmax=15, alpha=0.5, show_name=True, labfs=15):
    ylabs = []
    if show_name:
        for df in sig_dfs:
            for l in list(df['name']):
                if l not in ylabs:
                    ylabs.append(l)
        lab_to_int = {v:i for i,v in enumerate(ylabs)}
        y_val = [lab_to_int[l] for df in sig_dfs for l in df['name']]
    else:
        for df in sig_dfs:
            for l in list(df['GO']):
                if l not in ylabs:
                    ylabs.append(l)
        lab_to_int = {v:i for i,v in enumerate(ylabs)}
        y_val = [lab_to_int[l] for df in sig_dfs for l in df['GO']]
    x_val = [[i] * len(v) for i,v in enumerate(sig_dfs)]
    x_val = [v for l in x_val for v in l]
    sz = np.array([s for df in sig_dfs for s in df['size']])
    col = [p for df in sig_dfs for p in df['FDR']]
    fig = plt.figure(figsize=fs)
    plt.style.use(style)
    for i,v in enumerate(sig_dfs):
        tx = [i] * len(v)
        if show_name:
            ty = [lab_to_int[l] for l in v['name']]
        else:
            ty = [lab_to_int[l] for l in v['GO']]
        tsz = v['size']
        tcol = v['FDR']
        plt.scatter(tx, ty, s=tsz*szf, c=tcol, cmap=cmaps[i], vmin=vmin, vmax=vmax, alpha=alpha)
    proxies = [plt.scatter([], [], s=sz*szf, c='k', alpha=alpha) for sz in gs_prox]
    plabs = [str(sz) for sz in gs_prox]
    plt.legend(proxies, plabs, title='Gene Set Size', loc=4)
    plt.xlabel('dimension', fontsize=labfs)
    plt.xticks(range(len(sig_dfs)), xlabs)
    plt.yticks(range(len(ylabs)), ylabs, fontsize=labfs)
    plt.show()

def overlap_coef(gs1, gs2):
    gs1 = [s.strip() for s in gs1.split(';')][:-1]
    gs2 = [s.strip() for s in gs2.split(';')][:-1]
    overlap = set(gs1) & set(gs2)
    return(len(set(gs1) & set(gs2)) / min([len(gs1), len(gs2)]))

def get_nx_mdf(df, th_oc=0.1):
    perm = [list(i) for i in list(combinations(list(df['GO']), r=2))]
    par = [p[0] for p in perm]
    ch = [p[1] for p in perm]
    oc = [overlap_coef(df.loc[df.GO==p,'genes'].to_string(index=False), df.loc[df.GO==c,'genes'].to_string(index=False)) for p,c in zip(par, ch)]
    mdf = pd.DataFrame({'node1':par,
                       'node2':ch,
                       'oc':oc})
    mdf = mdf.loc[mdf['oc']>th_oc]
    return mdf

def plot_enrichmap(sig_df, k=0.5, scale=0.5, alpha=0.6, cmap='Reds', ns_scale=10, es_scale=5, fs=(5,5), vmin=-5, vmax=15):
    edge_df = get_nx_mdf(sig_df)
    G = nx.Graph()
    for g in list(sig_df['GO']):
        G.add_node(g)
    for i in range(len(edge_df)):
        G.add_edge(edge_df.iloc[i]['node1'], edge_df.iloc[i]['node2'], weight=edge_df.iloc[i]['oc'])
    ns = np.array(sig_df['size'])*ns_scale
    col = list(sig_df['FDR'])
    es = np.array(edge_df['oc'])*es_scale
    fig = plt.figure(figsize=fs)
    nx.draw(G, node_size=ns, alpha=alpha,
            node_color=col, cmap=cmap,
            vmin=vmin, vmax=vmax,
            width=es, with_labels=True,
            pos=nx.spring_layout(G, k=k, scale=scale))
    plt.show()

def plot_enrichmap_compare(df, col_dict, k=0.5, scale=0.5, alpha=0.6, ns_scale=10, es_scale=5, fs=(8,8), vmin=-5, vmax=15):
    ### provide sig_nodup_df
    edge_df = get_nx_mdf(df, th_oc=0.1)
    G = nx.Graph()
    for g in list(df['GO']):
        G.add_node(g)
    for i in range(len(edge_df)):
        G.add_edge(edge_df.iloc[i]['node1'], edge_df.iloc[i]['node2'], weight=edge_df.iloc[i]['oc'])
    ns = np.array(df['size'])*ns_scale
    col = [col_dict[d] for d in df['dimension']]
    es = np.array(edge_df['oc'])*es_scale
    fig = plt.figure(figsize=fs)
    nx.draw(G, node_size=ns, alpha=alpha,
            node_color=col,
            width=es, with_labels=True,
            pos=nx.spring_layout(G, k=k, scale=scale))
    plt.show()

def combine_sig_dfs(dfs, dims):
    ### dfs: list of sig_df; dims: list of corresponding sig_df dimension(str)
    tmp = []
    for df, d in zip(dfs, dims):
        df = df.assign(dimension=d)
        tmp.append(df)
    sig_df = pd.concat(tmp)
    sig_df = sig_df.reset_index(drop=True)
    sig_df = combine_dupGOrows(sig_df)
    return sig_df

def combine_dupGOrows(sig_df):
    dup_df = sig_df.loc[sig_df.duplicated('GO', keep=False)]
    no_dup_df = sig_df.loc[~sig_df.duplicated('GO', keep=False)]
    for go in set(dup_df['GO']):
        tdf = dup_df.loc[dup_df['GO']==go]
        tname = list(tdf['name'])[0]
        tgenes = ''.join(tdf['genes'].ravel())
        tgenes = [g.strip() for g in tgenes.split(';')][:-1]
        tgenes = list(set(tgenes))
        tgs = len(tgenes)
        tgenes = ';'.join(tgenes)
        tgenes = ''.join([tgenes,';'])
        no_dup_df = no_dup_df.append({
            'FDR':np.nan,
            'GO':go,
            'genes':tgenes,
            'name':tname,
            'pval':np.nan,
            'size':tgs,
            'dimension':np.nan}, ignore_index=True)
    return no_dup_df

def _plot_diff_semb_anno(d1, d2, polar_idx, text, th):
    diff_mean, diff_std = get_diff_mean_std(d1, d2)
    diff_idx = get_diff_idx(diff_mean, th=th)
    sig_texts = [text[i] for i in diff_idx]
    sig_idxs = [polar_idx[i] for i in diff_idx]
    sig_mean = [diff_mean[i] for i in diff_idx]

    fig = plt.figure(figsize=(15,15))
    plt.style.use('seaborn')
    ax = plt.subplot(111, projection='polar')
    plt.scatter(polar_idx, diff_mean, s = diff_std*500, alpha=0.5,
                c=diff_mean, cmap='OrRd')
    for g, x, y in zip(sig_texts, sig_idxs, sig_mean):
        plt.text(x,y,g, color='firebrick', fontsize=10, alpha=0.8)
    plt.show()

def _plot_diff_gemb_anno(d1, d2, polar_idx, genes, th):
    diff_mean, diff_std = get_diff_mean_std(d1, d2)
    diff_idx = get_diff_idx(diff_mean, th=th)
    sig_texts = genes[diff_idx]
    sig_idxs = [polar_idx[i] for i in diff_idx]
    sig_mean = [diff_mean[i] for i in diff_idx]

    fig = plt.figure(figsize=(15,15))
    plt.style.use('seaborn')
    ax = plt.subplot(111, projection='polar')
    plt.scatter(polar_idx, diff_mean, s = diff_std*500, alpha=0.5,
                c=diff_mean, cmap='OrRd')
    for g, x, y in zip(sig_texts, sig_idxs, sig_mean):
        plt.text(x,y,g, color='firebrick', fontsize=10, alpha=0.8)
    plt.show()

def _plot_diff_emb(d1, d2, polar_idx):
    diff_mean, diff_std = get_diff_mean_std(d1, d2)
    fig = plt.figure(figsize=(15,15))
    plt.style.use('seaborn')
    ax = plt.subplot(111, projection='polar')
    plt.scatter(polar_idx, diff_mean, s = diff_std*500, alpha=0.5,
                c=diff_mean, cmap='OrRd')
    plt.show()

def _get_diff_idx(arr, th):
    return [i for i, x in enumerate(arr) if x > th]

def _idx_to_polar(idxs):
    f = 360/len(idxs)
    return [i * f for i in idxs]

def _get_diff_mean_std(d1, d2):
    diff_dist = np.absolute(d1 - d2)
    print(diff_dist.min(), diff_dist.max(),
         diff_dist.mean(), diff_dist.std())
    diff_mean = np.mean(diff_dist, axis=0)
    diff_std = np.std(diff_dist, axis=0)
    print(diff_mean.shape, diff_std.shape)
    return diff_mean, diff_std
