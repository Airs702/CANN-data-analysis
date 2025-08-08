from matplotlib import  pyplot as plt, gridspec as grd
import numba
import numpy as np
from ripser import ripser
from scipy import  signal
from scipy.ndimage import gaussian_filter,  gaussian_filter1d,  binary_closing
from scipy.stats import  multivariate_normal
from scipy.spatial.distance import  pdist, squareform
import scipy.stats
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from sklearn import preprocessing
import os
from matplotlib import cm
import imageio
from scipy.stats import binned_statistic_2d
import traceback
import multiprocessing as mp
from tqdm import tqdm
import time


def get_spikes(spike_path, res=100000, dt=1000, sigma=5000,
               smooth0=True, speed0=True, min_speed=2.5):
    """
    Load and preprocess spike train data from npz file.

    This function converts raw spike times into a time-binned spike matrix, optionally applying Gaussian smoothing and filtering based on animal movement speed.

    Parameters:
        spike_path (str): Path to .npz file containing 'spike', 't', and optionally 'x', 'y'.
        res (int): Temporal resolution scaling factor (e.g., 100000 Hz).
        dt (int): Bin size in microseconds for discretizing spike trains.
        sigma (int): Standard deviation for Gaussian smoothing (in microseconds).
        smooth0 (bool): Whether to apply temporal smoothing to spike trains.
        speed0 (bool): Whether to apply speed filtering using x, y trajectories.
        min_speed (float): Minimum speed threshold for valid data (in cm/s).

    Returns:
        spikes_bin (ndarray): Binned and optionally smoothed spike matrix of shape (T, N).
        xx (ndarray, optional): X coordinates (if speed0=True).
        yy (ndarray, optional): Y coordinates (if speed0=True).
        tt (ndarray, optional): Time points (if speed0=True).
    """
    f = np.load(spike_path, allow_pickle=True)
    spikes_all = f['spike'][()]  # spike_train
    t = f['t']

    if speed0:
        x = f['x']
        y = f['y']

    min_time0 = np.min(t)
    max_time0 = np.max(t)

    # 提取 spike 区间
    cell_inds = np.arange(len(spikes_all))
    spikes = {}
    for i, m in enumerate(cell_inds):
        s = spikes_all[m]
        spikes[i] = np.array(s[(s >= min_time0) & (s < max_time0)])


    min_time = min_time0 * res
    max_time = max_time0 * res
    tt = np.arange(np.floor(min_time), np.ceil(max_time) + 1, dt)

    if smooth0:
        thresh = sigma * 5
        num_thresh = int(thresh / dt)
        num2_thresh = int(2 * num_thresh)
        sig2 = 1 / (2 * (sigma / res) ** 2)
        ker = np.exp(-np.power(np.arange(thresh + 1) / res, 2) * sig2)
        kerwhere = np.arange(-num_thresh, num_thresh) * dt

        spikes_bin = np.zeros((len(tt) + num2_thresh, len(spikes)))
        for n in spikes:
            spike_times = np.array(spikes[n] * res - min_time, dtype=int)
            spike_times = spike_times[(spike_times < (max_time - min_time)) & (spike_times > 0)]
            spikes_mod = dt - spike_times % dt
            spike_times = np.array(spike_times / dt, int)
            for m, j in enumerate(spike_times):
                spikes_bin[j:j + num2_thresh, n] += ker[np.abs(kerwhere + spikes_mod[m])]
        spikes_bin = spikes_bin[num_thresh - 1:-(num_thresh + 1), :]
        spikes_bin *= 1 / np.sqrt(2 * np.pi * (sigma / res) ** 2)
    else:
        spikes_bin = np.zeros((len(tt), len(spikes)), dtype=int)
        for n in spikes:
            spike_times = np.array(spikes[n] * res - min_time, dtype=int)
            spike_times = spike_times[(spike_times < (max_time - min_time)) & (spike_times > 0)]
            spike_times = np.array(spike_times / dt, int)
            for j in spike_times:
                spikes_bin[j, n] += 1

    # === speed filtering
    if speed0:
        xx, yy, tt_pos, speed = load_pos(spike_path)
        valid = speed > min_speed
        spikes_bin = spikes_bin[valid, :]
        xx = xx[valid]
        yy = yy[valid]
        tt = tt_pos[valid]
        return spikes_bin, xx, yy, tt

    return spikes_bin

##计算速度向量用于做速度过滤器
def load_pos(spike_path,res = 100000,dt = 1000):
    """
    Compute animal position and speed from spike data file.

    Interpolates animal positions to match spike time bins and computes smoothed velocity vectors and speed.

    Parameters:
        spike_path (str): Path to .npz file containing 't', 'x', 'y'.
        res (int): Time scaling factor to align with spike resolution.
        dt (int): Temporal bin size in microseconds.

    Returns:
        xx (ndarray): Interpolated x positions.
        yy (ndarray): Interpolated y positions.
        tt (ndarray): Corresponding time points (in seconds).
        speed (ndarray): Speed at each time point (in cm/s).
    """
    f = np.load(spike_path, allow_pickle=True)
    t = f['t']
    x = f['x']
    y = f['y']
    f.close()

    min_time0 = np.min(t)
    max_time0 = np.max(t)

    times = np.where((t >= min_time0) & (t < max_time0))
    x = x[times]
    y = y[times]
    t = t[times]

    min_time = min_time0 * res
    max_time = max_time0 * res

    tt = np.arange(np.floor(min_time), np.ceil(max_time) + 1, dt) / res

    idt = np.concatenate(([0], np.digitize(t[1:-1], tt[:]) - 1, [len(tt) + 1]))
    idtt = np.digitize(np.arange(len(tt)), idt) - 1

    idx = np.concatenate((np.unique(idtt), [np.max(idtt) + 1]))
    divisor = np.bincount(idtt)
    steps = (1.0 / divisor[divisor > 0])
    N = np.max(divisor)
    ranges = np.multiply(np.arange(N)[np.newaxis, :], steps[:, np.newaxis])
    ranges[ranges >= 1] = np.nan

    rangesx = x[idx[:-1], np.newaxis] + np.multiply(ranges, (x[idx[1:]] - x[idx[:-1]])[:, np.newaxis])
    xx = rangesx[~np.isnan(ranges)]

    rangesy = y[idx[:-1], np.newaxis] + np.multiply(ranges, (y[idx[1:]] - y[idx[:-1]])[:, np.newaxis])
    yy = rangesy[~np.isnan(ranges)]

    xxs = gaussian_filter1d(xx - np.min(xx), sigma=100)
    yys = gaussian_filter1d(yy - np.min(yy), sigma=100)
    dx = (xxs[1:] - xxs[:-1]) * 100
    dy = (yys[1:] - yys[:-1]) * 100
    speed = np.sqrt(dx ** 2 + dy ** 2) / 0.01
    speed = np.concatenate(([speed[0]], speed))
    return xx, yy, tt, speed

##用于在TDA前给高维点云数据的简单降维，默认降维到6维
def pca(data, dim=2):
    """
    Perform PCA (Principal Component Analysis) for dimensionality reduction.

    Parameters:
        data (ndarray): Input data matrix of shape (N_samples, N_features).
        dim (int): Target dimension for PCA projection.

    Returns:
        components (ndarray): Projected data of shape (N_samples, dim).
        var_exp (list): Variance explained by each principal component.
        evals (ndarray): Eigenvalues corresponding to the selected components.
    """
    if dim < 2:
        return data, [0]
    m, n = data.shape
    # mean center the data
    # data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eig(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dim]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    tot = np.sum(evals)
    var_exp = [(i / tot) * 100 for i in sorted(evals[:dim], reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    components = np.dot(evecs.T, data.T).T
    return components, var_exp, evals[:dim]


def sample_denoising(data, k=10, num_sample=500, omega=0.2, metric='euclidean'):
    """
    Perform denoising and greedy sampling based on mutual k-NN graph.

    Parameters:
        data (ndarray): High-dimensional point cloud data.
        k (int): Number of neighbors for local density estimation.
        num_sample (int): Number of samples to retain.
        omega (float): Suppression factor during greedy sampling.
        metric (str): Distance metric used for kNN ('euclidean', 'cosine', etc).

    Returns:
        inds (ndarray): Indices of sampled points.
        d (ndarray): Pairwise similarity matrix of sampled points.
        Fs (ndarray): Sampling scores at each step.
    """
    n = data.shape[0]
    leftinds = np.arange(n)
    F_D = np.zeros(n)
    if metric in ("cosine", "correlation", "dice", "jaccard"):
        angular = True
    else:
        angular = False

    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :k]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    sigmas, rhos = smooth_knn_dist(knn_dists, k, local_connectivity=0)
    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()
    X = result.toarray()
    F = np.sum(X, 1)
    Fs = np.zeros(num_sample)
    Fs[0] = np.max(F)
    i = np.argmax(F)
    inds_all = np.arange(n)
    inds_left = inds_all > -1
    inds_left[i] = False
    inds = np.zeros(num_sample, dtype=int)
    inds[0] = i
    for j in np.arange(1, num_sample):
        F -= omega * X[i, :]
        Fmax = np.argmax(F[inds_left])
        Fs[j] = F[Fmax]
        i = inds_all[inds_left][Fmax]

        inds_left[i] = False
        inds[j] = i
    d = np.zeros((num_sample, num_sample))

    for j, i in enumerate(inds):
        d[j, :] = X[i, inds]
    return inds, d, Fs

@numba.njit(
    fastmath=True
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=0.0, bandwidth=1.0):
    """
    Compute smoothed local distances for kNN graph with entropy balancing.

    Parameters:
        distances (ndarray): kNN distance matrix.
        k (int): Number of neighbors.
        n_iter (int): Number of binary search iterations.
        local_connectivity (float): Minimum local connectivity.
        bandwidth (float): Bandwidth parameter.

    Returns:
        sigmas (ndarray): Smoothed sigma values for each point.
        rhos (ndarray): Minimum distances (connectivity cutoff) for each point.
    """
    target = np.log2(k) * bandwidth
    #    target = np.log(k) * bandwidth
    #    target = k

    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = np.inf
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > 1e-5:
                    rho[i] += interpolation * (
                            non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                #                    psum += d / mid

                else:
                    psum += 1.0
            #                    psum += 0

            if np.fabs(psum - target) < 1e-5:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == np.inf:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        result[i] = mid
        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < 1e-3 * mean_ith_distances:
                result[i] = 1e-3 * mean_ith_distances
        else:
            if result[i] < 1e-3 * mean_distances:
                result[i] = 1e-3 * mean_distances

    return result, rho

def second_build(data, indstemp, nbs=800, metric='cosine'):
    """
    Reconstruct distance matrix after denoising for persistent homology.

    Parameters:
        data (ndarray): PCA-reduced data matrix.
        indstemp (ndarray): Indices of sampled points.
        nbs (int): Number of neighbors in reconstructed graph.
        metric (str): Distance metric ('cosine', 'euclidean', etc).

    Returns:
        d (ndarray): Symmetric distance matrix used for persistent homology.
    """
    # Filter the data using the sampled point indices
    data = data[indstemp, :]

    # Compute the pairwise distance matrix
    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :nbs]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    # Compute smoothed kernel widths
    sigmas, rhos = smooth_knn_dist(knn_dists, nbs, local_connectivity=0)
    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)

    # Construct a sparse graph
    result = coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()

    # Build the final distance matrix
    d = result.toarray()
    d = -np.log(d)
    np.fill_diagonal(d, 0)

    return d

@numba.njit(parallel=True, fastmath=True) 
def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """
    Compute membership strength matrix from smoothed kNN graph.

    Parameters:
        knn_indices (ndarray): Indices of k-nearest neighbors.
        knn_dists (ndarray): Corresponding distances.
        sigmas (ndarray): Local bandwidths.
        rhos (ndarray): Minimum distance thresholds.

    Returns:
        rows (ndarray): Row indices for sparse matrix.
        cols (ndarray): Column indices for sparse matrix.
        vals (ndarray): Weight values for sparse matrix.
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    rows = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_samples * n_neighbors), dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
                #val = ((knn_dists[i, j] - rhos[i]) / (sigmas[i]))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals

##对TDA后的结果进行可视化
def plot_barcode(persistence):
    """
    Plot barcode diagram from persistent homology result.

    Parameters:
        persistence (list of ndarray): List of birth-death pairs per homology dimension.
    """
    cs = np.repeat([[0, 0.55, 0.2]], 3).reshape(3, 3).T  # RGB color for each dimension
    alpha = 1
    inf_delta = 0.1
    colormap = cs
    maxdim = len(persistence) - 1
    dims = np.arange(maxdim + 1)
    labels = ["$H_0$", "$H_1$", "$H_2$"]

    # Determine axis range
    min_birth, max_death = 0, 0
    for dim in dims:
        persistence_dim = persistence[dim][~np.isinf(persistence[dim][:, 1]), :]
        if persistence_dim.size > 0:
            min_birth = min(min_birth, np.min(persistence_dim))
            max_death = max(max_death, np.max(persistence_dim))

    delta = (max_death - min_birth) * inf_delta
    infinity = max_death + delta
    axis_start = min_birth - delta

    # Create plot
    fig = plt.figure(figsize=(10, 6))
    gs = grd.GridSpec(len(dims), 1)

    for dim in dims:
        axes = plt.subplot(gs[dim])
        axes.axis('on')
        axes.set_yticks([])
        axes.set_ylabel(labels[dim], rotation=0, labelpad=20, fontsize=12)

        d = np.copy(persistence[dim])
        d[np.isinf(d[:, 1]), 1] = infinity
        dlife = (d[:, 1] - d[:, 0])

        # Select top 30 bars by lifetime
        dinds = np.argsort(dlife)[-30:]
        if dim > 0:
            dinds = dinds[np.flip(np.argsort(d[dinds, 0]))]

        axes.barh(
            0.5 + np.arange(len(dinds)),
            dlife[dinds],
            height=0.8,
            left=d[dinds, 0],
            alpha=alpha,
            color=colormap[dim],
            linewidth=0,
        )

        axes.plot([0, 0], [0, len(dinds)], c='k', linestyle='-', lw=1)
        axes.plot([0, len(dinds)], [0, 0], c='k', linestyle='-', lw=1)
        axes.set_xlim([axis_start, infinity])

def TDAvis(spikes,
           dim=6,
           num_times=5,
           active_times=15000,
           k=1000,
           n_points=1200,
           metric='cosine',
           nbs=800,
           maxdim=1,
           coeff=47,
           show=True):
    """
    High-level function: Load spike data → PCA → Denoising → Second Build → TDA → Visualization

    Parameters:
    ----------
    spikes : np.ndarray
        Preprocessed spike matrix of shape (T, N), where T is time and N is number of neurons

    dim : int
        PCA target dimension
    num_times : int
        Step size for downsampling timepoints
    active_times : int
        Number of most active timepoints to keep
    k : int
        KNN for denoising
    n_points : int
        Number of sampled points after denoising
    metric : str
        Distance metric (e.g., 'euclidean', 'cosine')
    nbs : int
        Number of neighbors for second build
    maxdim : int
        Maximum homology dimension for ripser
    coeff : int
        Coefficient field for persistent homology
    show : bool
        Whether to plot the barcode

    Returns:
    -------
    persistence : dict
        Dictionary containing persistence diagrams
    """
    num_neurons = len(spikes[0, :])
    print(num_neurons)

    times_cube = np.arange(0, spikes.shape[0], num_times)
    movetimes = np.sort(np.argsort(np.sum(spikes[times_cube, :], 1))[-active_times:])
    movetimes = times_cube[movetimes]

    dimred, *_ = pca(preprocessing.scale(spikes[movetimes, :]), dim=dim)

    indstemp, *_ = sample_denoising(dimred, k, n_points, 1, metric)
    d = second_build(dimred, indstemp, metric=metric, nbs=nbs)
    np.fill_diagonal(d, 0)

    persistence = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles=True, distance_matrix=True)

    if show:
        plot_barcode(persistence['dgms'])
        plt.show()

        # 保存文件
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed(
        'Results/spikes_persistence.npz',
        persistence=persistence,
        indstemp=indstemp,
        movetimes=movetimes,
        n_points=n_points,

    )

    return persistence

##对TDA后的结果进行解码，将bump的位置解码到圆环上
def decode_circular_coordinates(persistence_path,
           spike_path,
           real_ground = True,
           real_of = True):
    """
    Decode circular coordinates (bump positions) from cohomology.

    Parameters:
        persistence_path (str): Path to saved npz from TDAvis.
        spike_path (str): Path to spike data file.
        real_ground (bool): Whether x, y, t ground truth exists.
        real_of (bool): Whether experiment was performed in open field.

    Returns:
        decoding_path (str): Path to saved decoding results (.npz).
    """
    ph_classes = [0, 1]  # Decode the ith most persistent cohomology class
    num_circ = len(ph_classes)
    dec_tresh = 0.99
    coeff = 47
    data = np.load(persistence_path, allow_pickle=True)
    persistence = data['persistence'].item()
    indstemp = data['indstemp']
    movetimes = data['movetimes']
    n_points = data['n_points']
    diagrams = persistence["dgms"]  # the multiset describing the lives of the persistence classes
    cocycles = persistence["cocycles"][1]  # the cocycle representatives for the 1-dim classes
    dists_land = persistence["dperm2all"]  # the pairwise distance between the points
    births1 = diagrams[1][:, 0]  # the time of birth for the 1-dim classes
    deaths1 = diagrams[1][:, 1]  # the time of death for the 1-dim classes
    deaths1[np.isinf(deaths1)] = 0
    lives1 = deaths1 - births1  # the lifetime for the 1-dim classes
    iMax = np.argsort(lives1)
    coords1 = np.zeros((num_circ, len(indstemp)))
    threshold = births1[iMax[-2]] + (deaths1[iMax[-2]] - births1[iMax[-2]]) * dec_tresh


    for c in ph_classes:
        cocycle = cocycles[iMax[-(c + 1)]]
        coords1[c, :], inds = get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)

    if real_ground:#用户所提供的数据是否有真实的xyt
        sspikes, xx, yy, tt = get_spikes(spike_path,smooth0=True, speed0=True)
    else:
        sspikes = get_spikes(spike_path,smooth0=True, speed0=True)
    num_neurons = len(sspikes[0, :])
    centcosall = np.zeros((num_neurons, 2, n_points))
    centsinall = np.zeros((num_neurons, 2, n_points))
    dspk = preprocessing.scale(sspikes[movetimes[indstemp], :])

    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        centcosall[neurid, :, :] = np.multiply(np.cos(coords1[:, :] * 2 * np.pi), spktemp)
        centsinall[neurid, :, :] = np.multiply(np.sin(coords1[:, :] * 2 * np.pi), spktemp)

    if real_ground:#用户所提供的数据是否有真实的xyt
        sspikes, xx, yy, tt = get_spikes(spike_path,smooth0=True, speed0=True)
        spikes, __, __, __ = get_spikes(spike_path,smooth0=False, speed0=True)
    else:
        sspikes = get_spikes(spike_path,smooth0=True, speed0=False)
        spikes = get_spikes(spike_path,smooth0=False, speed0=False)

    times = np.where(np.sum(spikes > 0, 1) >= 1)[0]
    dspk = preprocessing.scale(sspikes)
    sspikes = sspikes[times, :]
    dspk = dspk[times, :]

    a = np.zeros((len(sspikes[:, 0]), 2, num_neurons))
    for n in range(num_neurons):
        a[:, :, n] = np.multiply(dspk[:, n:n + 1], np.sum(centcosall[n, :, :], 1))

    c = np.zeros((len(sspikes[:, 0]), 2, num_neurons))
    for n in range(num_neurons):
        c[:, :, n] = np.multiply(dspk[:, n:n + 1], np.sum(centsinall[n, :, :], 1))

    mtot2 = np.sum(c, 2)
    mtot1 = np.sum(a, 2)
    coords = np.arctan2(mtot2, mtot1) % (2 * np.pi)
    if real_of:#用户的数据是否是来自真实的OF场地
        coordsbox = coords.copy()
        times_box = times.copy()
    else:
        sspikes, xx, yy, tt = get_spikes(spike_path,smooth0=True, speed0=True)
        spikes, __, __, __ = get_spikes(spike_path,smooth0=False, speed0=True)
        dspk = preprocessing.scale(sspikes)
        times_box = np.where(np.sum(spikes > 0, 1) >= 1)[0]
        dspk = dspk[times_box, :]

        a = np.zeros((len(times_box), 2, num_neurons))
        for n in range(num_neurons):
            a[:, :, n] = np.multiply(dspk[:, n:n + 1], np.sum(centcosall[n, :, :], 1))

        c = np.zeros((len(times_box), 2, num_neurons))
        for n in range(num_neurons):
            c[:, :, n] = np.multiply(dspk[:, n:n + 1], np.sum(centsinall[n, :, :], 1))

        mtot2 = np.sum(c, 2)
        mtot1 = np.sum(a, 2)
        coordsbox = np.arctan2(mtot2, mtot1) % (2 * np.pi)
        # 保存解码结果
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed(
        'Results/spikes_decoding.npz',
        coords=coords,
        coordsbox=coordsbox,
        times=times,
        times_box=times_box,
        centcosall=centcosall,
        centsinall=centsinall
    )
    return 'Results/spikes_decoding.npz'


def get_coords(cocycle, threshold, num_sampled, dists, coeff):
    """
    Reconstruct circular coordinates from cocycle information.

    Parameters:
        cocycle (ndarray): Persistent cocycle representative.
        threshold (float): Maximum allowable edge distance.
        num_sampled (int): Number of sampled points.
        dists (ndarray): Pairwise distance matrix.
        coeff (int): Finite field modulus for cohomology.

    Returns:
        f (ndarray): Circular coordinate values (in [0,1]).
        verts (ndarray): Indices of used vertices.
    """
    zint = np.where(coeff - cocycle[:, 2] < cocycle[:, 2])
    cocycle[zint, 2] = cocycle[zint, 2] - coeff
    d = np.zeros((num_sampled, num_sampled))
    d[np.tril_indices(num_sampled)] = np.NaN
    d[cocycle[:, 1], cocycle[:, 0]] = cocycle[:, 2]
    d[dists > threshold] = np.NaN
    d[dists == 0] = np.NaN
    edges = np.where(~np.isnan(d))
    verts = np.array(np.unique(edges))
    num_edges = np.shape(edges)[1]
    num_verts = np.size(verts)
    values = d[edges]
    A = np.zeros((num_edges, num_verts), dtype=int)
    v1 = np.zeros((num_edges, 2), dtype=int)
    v2 = np.zeros((num_edges, 2), dtype=int)
    for i in range(num_edges):
        v1[i, :] = [i, np.where(verts == edges[0][i])[0]]
        v2[i, :] = [i, np.where(verts == edges[1][i])[0]]

    A[v1[:, 0], v1[:, 1]] = -1
    A[v2[:, 0], v2[:, 1]] = 1

    L = np.ones((num_edges,))
    Aw = A * np.sqrt(L[:, np.newaxis])
    Bw = values * np.sqrt(L)
    f = lsmr(Aw, Bw)[0] % 1
    return f, verts

def smooth_tuning_map(mtot, numangsint, sig, bClose = True):
    """
    Smooth activity map over circular topology (e.g., torus).

    Parameters:
        mtot (ndarray): Raw activity map matrix.
        numangsint (int): Grid resolution.
        sig (float): Smoothing kernel standard deviation.
        bClose (bool): Whether to assume circular boundary conditions.

    Returns:
        mtot_out (ndarray): Smoothed map matrix.
    """
    numangsint_1 = numangsint-1
    mid = int((numangsint_1)/2)
    indstemp1 = np.zeros((numangsint_1,numangsint_1), dtype=int)
    indstemp1[indstemp1==0] = np.arange((numangsint_1)**2)
    indstemp1temp = indstemp1.copy()
    mid = int((numangsint_1)/2)
    mtemp1_3 = mtot.copy()
    for i in range(numangsint_1):
        mtemp1_3[i,:] = np.roll(mtemp1_3[i,:],int(i/2))
    mtot_out = np.zeros_like(mtot)
    mtemp1_4 = np.concatenate((mtemp1_3, mtemp1_3, mtemp1_3),1)
    mtemp1_5 = np.zeros_like(mtemp1_4)
    mtemp1_5[:, :mid] = mtemp1_4[:, (numangsint_1)*3-mid:]
    mtemp1_5[:, mid:] = mtemp1_4[:,:(numangsint_1)*3-mid]
    if bClose:
        mtemp1_6 = smooth_image(np.concatenate((mtemp1_5,mtemp1_4,mtemp1_5)) ,sigma = sig)
    else:
        mtemp1_6 = gaussian_filter(np.concatenate((mtemp1_5,mtemp1_4,mtemp1_5)) ,sigma = sig)
    for i in range(numangsint_1):
        mtot_out[i, :] = mtemp1_6[(numangsint_1)+i,
                                          (numangsint_1) + (int(i/2) +1):(numangsint_1)*2 + (int(i/2) +1)]
    return mtot_out

def smooth_image(img, sigma):
    """
    Smooth image using multivariate Gaussian kernel, handling missing (NaN) values.

    Parameters:
        img (ndarray): Input image matrix.
        sigma (float): Standard deviation of smoothing kernel.

    Returns:
        imgC (ndarray): Smoothed image with inpainting around NaNs.
    """
    filterSize = max(np.shape(img))
    grid = np.arange(-filterSize+1, filterSize, 1)
#    covariance = np.square([sigma, sigma])
    xx,yy = np.meshgrid(grid, grid)

    pos = np.dstack((xx, yy))

    var = multivariate_normal(mean=[0,0], cov=[[sigma**2,0],[0,sigma**2]])
    k = var.pdf(pos)
    k = k/np.sum(k)

    nans = np.isnan(img)
    imgA = img.copy()
    imgA[nans] = 0
    imgA = scipy.signal.convolve2d(imgA, k, mode='valid')
#    imgA = gaussian_filter(imgA, sigma = sigma, mode = mode)
    imgD = img.copy()
    imgD[nans] = 0
    imgD[~nans] = 1
    radius = 1
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    dk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=bool)
    imgE = np.zeros((filterSize+2,filterSize+2))
    imgE[1:-1,1:-1] = imgD
    imgE= binary_closing(imgE,iterations =1, structure =dk)
    imgD = imgE[1:-1,1:-1]

    imgB = np.divide(scipy.signal.convolve2d(imgD, k, mode='valid'), scipy.signal.convolve2d(np.ones(np.shape(imgD)), k, mode='valid'))
    imgC = np.divide(imgA,imgB)
    imgC[imgD==0] = -np.inf
    return imgC


def plot_3d_bump_on_torus(decoding_path, spike_path, output_path='torus_bump.gif',
                          numangsint=51, r1=1.5, r2=1.0, window_size=300,
                          frame_step=5, n_frames=20, fps=5):
    """
    Visualize the movement of the neural activity bump on a torus and generate an animated GIF.

    Parameters:
        decoding_path (str): Path to the .npz file containing 'coordsbox' and 'times_box'.
        spike_path (str): Path to the spike data file.
        output_path (str): Output path for the generated GIF.
        numangsint (int): Grid resolution for the torus surface.
        r1 (float): Major radius of the torus.
        r2 (float): Minor radius of the torus.
        window_size (int): Time window (in number of time points) for each frame.
        frame_step (int): Step size to slide the time window between frames.
        n_frames (int): Total number of frames in the animation.
        fps (int): Frames per second for the output GIF.
    """

    f = np.load(decoding_path, allow_pickle=True)
    coords = f['coordsbox']
    times = f['times_box']
    f.close()

    spk, *_ = get_spikes(spike_path, smooth0=False, speed0=True)

    frames = []
    prev_m = None

    for frame_idx in range(n_frames):
        start_idx = frame_idx * frame_step
        end_idx = start_idx + window_size
        if end_idx > np.max(times):
            break

        mask = (times >= start_idx) & (times < end_idx)
        coords_window = coords[mask]
        if len(coords_window) == 0:
            continue

        spk_window = spk[times[mask], :]
        activity = np.sum(spk_window, axis=1)

        m, x_edge, y_edge, _ = binned_statistic_2d(
            coords_window[:, 0], coords_window[:, 1],
            activity,
            statistic='sum',
            bins=np.linspace(0, 2 * np.pi, numangsint - 1)
        )
        m = np.nan_to_num(m)
        m = smooth_tuning_map(m, numangsint - 1, sig=4.0, bClose=True)
        m = gaussian_filter(m, sigma=1.0)

        if prev_m is not None:
            m = 0.7 * prev_m + 0.3 * m
        prev_m = m

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(x_edge, y_edge)
        X = (X + np.pi / 5) % (2 * np.pi)
        x = (r1 + r2 * np.cos(X)) * np.cos(Y)
        y = (r1 + r2 * np.cos(X)) * np.sin(Y)
        z = r2 * np.sin(X)

        ax.plot_surface(
            x, y, z,
            facecolors=cm.viridis(m / (np.max(m) + 1e-9)),
            alpha=1, linewidth=0.1, antialiased=True,
            rstride=1, cstride=1, shade=False
        )
        ax.set_zlim(-2, 2)
        ax.view_init(-125, 135)
        ax.axis('off')

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(image)
        plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps)
    print(f"✅ GIF saved as {output_path}")


def shuffle_spike_trains(sspikes):
    """对尖峰序列进行随机循环移位"""
    shuffled = sspikes.copy()
    num_neurons = shuffled.shape[1]

    # 每个神经元独立移位
    for n in range(num_neurons):
        shift = np.random.randint(0, shuffled.shape[0]*0.1)
        shuffled[:, n] = np.roll(shuffled[:, n], shift)

    return shuffled


def compute_persistence(sspikes, dim=6, num_times=5, active_times=15000,
                        k=1000, n_points=1200, metric='cosine',
                        nbs=800, maxdim=1, coeff=47):
    """
    计算持久同源，添加时间记录
    """
    start_time = time.time()

    # 时间点下采样
    times_cube = np.arange(0, sspikes.shape[0], num_times)

    # 选择最活跃的时间点
    movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube, :], 1))[-active_times:])
    movetimes = times_cube[movetimes]

    # PCA降维
    scaled_data = preprocessing.scale(sspikes[movetimes, :])
    dimred, *_ = pca(scaled_data, dim=dim)

    # 点云采样（降噪）
    indstemp, *_ = sample_denoising(dimred, k, n_points, 1, metric)

    # 构建距离矩阵
    d = second_build(dimred, indstemp, metric=metric, nbs=nbs)
    np.fill_diagonal(d, 0)

    # 计算持久同源
    persistence = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles=True, distance_matrix=True)

    calc_time = time.time() - start_time
    print(f"Persistent homology computation completed - Time: {calc_time:.1f}s | Points: {len(d)}")

    return persistence


def process_single_shuffle(args):
    """处理单个shuffle任务"""
    i, sspikes, kwargs = args
    try:
        shuffled_data = shuffle_spike_trains(sspikes)
        persistence = compute_persistence(shuffled_data, **kwargs)

        dim_max_lifetimes = {}
        for dim in [0, 1, 2]:
            if dim < len(persistence['dgms']):
                # 过滤掉无限值
                valid_bars = [bar for bar in persistence['dgms'][dim] if not np.isinf(bar[1])]
                if valid_bars:
                    lifetimes = [bar[1] - bar[0] for bar in valid_bars]
                    if lifetimes:
                        dim_max_lifetimes[dim] = max(lifetimes)
        return dim_max_lifetimes
    except Exception as e:
        print(f"Shuffle {i} 失败: {str(e)}")
        return {}


def run_shuffle_analysis(sspikes, num_shuffles=1000, num_cores=4,**kwargs):
    """执行shuffle分析，使用多进程并行"""
    max_lifetimes = {0: [], 1: [], 2: []}
    start_time = time.time()

    # 预估单次迭代时间（先运行一次）
    print("Running test iteration to estimate runtime...")

    test_start = time.time()
    _ = process_single_shuffle((0, sspikes, kwargs))
    test_time = time.time() - test_start

    # 打印预估信息
    total_est = test_time * num_shuffles
    print(f"Estimated total runtime: {total_est / 60:.1f} minutes ({total_est / 3600:.1f} hours)")

    print(f"Average time per iteration: {test_time:.1f} seconds")

    print(f"Starting {num_shuffles} shuffle iterations using {num_cores} processes...")


    # 准备任务列表
    tasks = [(i, sspikes, kwargs) for i in range(num_shuffles)]

    # 使用多进程池并行处理
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(process_single_shuffle, tasks),
                            total=num_shuffles,
                            desc="Running shuffle analysis"))

    # 收集结果
    for res in results:
        for dim, lifetime in res.items():
            max_lifetimes[dim].append(lifetime)

    total_time = time.time() - start_time
    print(f"Completed! Total elapsed time: {total_time / 60:.1f} minutes")
    print(f"Average time per iteration: {total_time / num_shuffles:.1f} seconds")

    return max_lifetimes


def TDAvis_shuffle(spike_path=None, sspikes=None, dim=6, num_times=5, active_times=15000,
           k=1000, n_points=1200, metric='cosine', nbs=800, maxdim=1,
           coeff=47, show=True, do_shuffle=False, num_shuffles=1000,num_cores=4):
    """
    增强版TDAvis函数
    """
    start_time = time.time()

    # 加载数据
    if sspikes is None:
        print(f"Loading data from: {spike_path}")
        load_start = time.time()
        sspikes, *_ = get_spikes(spike_path)
        print(
            f"Data loaded successfully — elapsed time: {time.time() - load_start:.1f} seconds | shape: {sspikes.shape}")

    # 计算真实数据的持久同源
    print("Computing persistent homology for real data...")

    try:
        real_persistence = compute_persistence(sspikes, dim, num_times, active_times,
                                               k, n_points, metric, nbs, maxdim, coeff)
    except Exception as e:
        print(f"Persistent homology computation failed: {str(e)}")

        traceback.print_exc()
        return None, None

    # 执行shuffle分析
    shuffle_max = None
    if do_shuffle:
        print(f"\nStarting shuffle analysis with {num_shuffles} iterations...")

        shuffle_max = run_shuffle_analysis(sspikes, num_shuffles, dim=dim, num_times=num_times,
                                           active_times=active_times, k=k, n_points=n_points,
                                           metric=metric, nbs=nbs, maxdim=maxdim, coeff=coeff)

        # 打印shuffle结果摘要
        print("\nSummary of shuffle-based analysis:")
        for dim in [0, 1, 2]:
            if shuffle_max and dim in shuffle_max and shuffle_max[dim]:
                print(f"H{dim}: {len(shuffle_max[dim])} valid iterations | "
                      f"Mean maximum persistence: {np.mean(shuffle_max[dim]):.4f} | "
                      f"99.9th percentile: {np.percentile(shuffle_max[dim], 99.9):.4f}")

    # 可视化
    if show:
        print("Generating barcode visualization...")
        # 确保shuffle_max不为None
        plot_barcode_with_shuffle(real_persistence, shuffle_max if shuffle_max is not None else {})
        plt.show()

    total_time = time.time() - start_time
    print(f"\nAnalysis completed. Total duration: {total_time / 60:.1f} minutes.")
    return real_persistence, shuffle_max


def plot_barcode_with_shuffle(persistence, shuffle_max):
    """
    绘制barcode并添加shuffle区域标记
    """
    # 处理shuffle_max为None的情况
    if shuffle_max is None:
        shuffle_max = {}


    cs = np.repeat([[0, 0.55, 0.2]], 3).reshape(3, 3).T
    alpha = 1
    inf_delta = 0.1
    colormap = cs
    maxdim = len(persistence['dgms']) - 1
    dims = np.arange(maxdim + 1)


    min_birth, max_death = 0, 0
    for dim in dims:
        # 过滤掉无限值
        valid_bars = [bar for bar in persistence['dgms'][dim] if not np.isinf(bar[1])]
        if valid_bars:
            min_birth = min(min_birth, np.min(valid_bars))
            max_death = max(max_death, np.max(valid_bars))

    # 处理没有有效条带的情况
    if max_death == 0 and min_birth == 0:
        min_birth = 0
        max_death = 1

    delta = (max_death - min_birth) * inf_delta
    infinity = max_death + delta

    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    gs = grd.GridSpec(len(dims), 1)

    # 获取shuffle阈值（每个维度99.9%分位数）
    thresholds = {}
    for dim in dims:
        if dim in shuffle_max and shuffle_max[dim]:
            thresholds[dim] = np.percentile(shuffle_max[dim], 99.9)
        else:
            thresholds[dim] = 0

    for dit, dim in enumerate(dims):
        axes = plt.subplot(gs[dim])
        axes.axis('off')

        # 添加灰色背景表示shuffle区域
        if dim in thresholds:
            axes.axvspan(0, thresholds[dim], alpha=0.2, color='gray', zorder=-3)
            axes.axvline(x=thresholds[dim], color='gray', linestyle='--', alpha=0.7)

        # 过滤掉无限值
        d = np.array([bar for bar in persistence['dgms'][dim] if not np.isinf(bar[1])])
        if len(d) == 0:
            d = np.zeros((0, 2))

        d = np.copy(d)
        d[np.isinf(d[:, 1]), 1] = infinity
        dlife = (d[:, 1] - d[:, 0])

        # 选择前30个最长寿命的条带
        if len(dlife) > 0:
            dinds = np.argsort(dlife)[-30:]
            if dim > 0:
                dinds = dinds[np.flip(np.argsort(d[dinds, 0]))]

            # 标记显著条带
            significant_bars = []
            for idx in dinds:
                if dlife[idx] > thresholds.get(dim, 0):
                    significant_bars.append(idx)

            # 绘制条带
            for i, idx in enumerate(dinds):
                color = 'red' if idx in significant_bars else colormap[dim]
                axes.barh(
                    0.5 + i,
                    dlife[idx],
                    height=0.8,
                    left=d[idx, 0],
                    alpha=alpha,
                    color=color,
                    linewidth=0,
                )

            indsall = len(dinds)
        else:
            indsall = 0

        axes.plot([0, 0], [0, indsall], c='k', linestyle='-', lw=1)
        axes.plot([0, indsall], [0, 0], c='k', linestyle='-', lw=1)
        axes.set_xlim([0, infinity])
        axes.set_title(f"$H_{dim}$", loc='left')

    plt.tight_layout()
    return fig