import os
import os.path
import sys
import math
import itertools as it
import time
import copy

import numpy as np
import scipy.interpolate

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import (DBSCAN, KMeans)
import sklearn.metrics as metrics
import sklearn.manifold as manifold

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib import offsetbox
import matplotlib.patches as mpatches
import matplotlib.text as mtext

def load_files(files):
    """
    Args:
        files (list of str): name of files from wich to extract datas.
            Excpected .npy files with one dictionary with 'FSD_SMAP'
            key word.
    Returns:
        (list of np.array) The matrix is shape of (n,3). Columns 0 and 1 represent
            a position in a visual field. Column 2 is a number of fixations.
    """
    data_list = []
    for data_file in files:
        try:
            data = np.load(data_file)
            data = data.item()
            data = data['FSD_SMAP']
            data_list.append(data)
        except Exception, ex:
            print "Error in {0}:".format(data_file)
            raise ex
    return data_list

def feature_extraction(file_datas):
    """
    Args:
        file_datas (list of np.array with (n,3) shape): expecting return value
            of load_files function.
    Returns:
        (np.matrix) 2D matrix. The row is a sample. The column is a feature.
    """
    data_list = []
    for data in file_datas:
        data = data[:,2]
        data_list.append(data/np.sum(data))
    return np.column_stack(data_list).T

def plot_pca(fg,ax,pca):
    points = np.sort(pca.explained_variance_ratio_)[::-1]

    plot_points(fg,ax,points)

    ax.set_title(
        "Variance ratio for PCA with {0} dimensions".format(pca.n_components_)
    )
    ax.set_xlabel("dimension")
    ax.set_ylabel("variance ratio")

def tick_step(points,max_ticks):
    i = 1
    while points > max_ticks * i:
        i *= 10
    return i

def plot_points(fg,ax,points):
    """"
    Args:
        fg (figure)
        ax (axes)
        points (vector): descending ordered floats
    """
    pts_len = len(points)
    ax.plot(np.arange(1,pts_len+1,1),points)

    ax.set_xlim([0,pts_len+1]) 
    ticks = range(0,pts_len+1,tick_step(pts_len,10))
    if len(ticks) < 3:
        ticks.append(pts_len+1)
    ax.set_xticks(ticks)
    ax.xaxis.set_minor_locator(ticker.IndexLocator(tick_step(pts_len,100),-1))
    ax.grid(which='major',alpha=0.9)
    ax.grid(which='minor',alpha=0.4)

def get_sorted_neighbors(data_matrix,n_neighbors):
    """
    Args:
        data_matrix (np.array): data matrix to analyze
        n_neighbors (int): number of neighbors for each smaple
    Rerturns:
        (weight matrix) with sorted values in rows
    """
    graph = kneighbors_graph(data_matrix,n_neighbors,mode='distance')
    graph = graph.toarray()
    for row in graph:
        row.sort()
    return graph

def dynamic_subplots(n_subplots):
    root = math.sqrt(n_subplots)
    x = int(math.ceil(root))
    y = int(math.ceil(n_subplots / float(x)))
    fg, ax = plt.subplots(x,y)
    return fg, np.array(ax).ravel()

def get_image_grid(smap):
    locations = smap[:,0:2]
    values = smap[:,2]
    min_x, min_y = np.subtract(np.min(locations,axis=0),1)
    max_x, max_y = np.add(np.max(locations,axis=0),1)
    mesh_X, mesh_Y = np.meshgrid(
        np.linspace(min_x, max_x, 100),
        np.linspace(min_y, max_y, 100),
        indexing='xy'
    )
    mesh_z = scipy.interpolate.griddata(
        locations, values, (mesh_X, mesh_Y), method='cubic', fill_value=0
    )
    return mesh_z, (min_x,max_x,min_y,max_y)
   
def get_offset_image(smap):
    return offsetbox.OffsetImage(
        get_image_grid(smap)[0], cmap=cm.coolwarm, zoom=0.3
    )
            

if __name__ == '__main__':
    import argparse

    def positive_int(argument):
        val = int(argument)
        if val < 1:
            raise argparse.ArgumentTypeError("{0} isn't positive int".format(val))
        return val

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+', type=str, help="files to analyze",
        metavar="FILE", dest='files')
    parser.add_argument('--pca-plot', action='store_true',
        help="plot pca variance ratio")
    parser.add_argument('--pca-set', action='store', type=str,
        help="set pca components")
    # DBSCAN analysis
    parser.add_argument('--dbscan-minpts-plot', action='store_true',
        help="plot mean distances")
    parser.add_argument('--dbscan-eps-plot', action='store_true',
        help="plot k-th neighbors distances")
    # Clustering analysis
    CLUSTER= ['k-means','dbscan']
    parser.add_argument('--cluster', action='store', type=str, choices=CLUSTER,
        help="choose clustering algorithm", metavar="CLUSTER", dest='cluster')
    parser.add_argument('--silhouette', action='store', nargs='?', default=None,
        help="plot silhouette analysis")
    ANALYSES = {'isomap', 'tsne', 'mds'}
    parser.add_argument('--cluster-analysis', action='store', type=str, choices=ANALYSES,
        help="cluster data", metavar="ANALISIS")
    # Other
    parser.add_argument('--out-dir', action='store', type=str,
        help="output dir name for cluster analysis", metavar='DIR')

    args = parser.parse_args()

### Load files ###
    files = args.files
    data_list = load_files(files)

### Feature extraction ###
    data_matrix = feature_extraction(data_list)

### Transformation ###
    scaler = StandardScaler()
    data_matrix = scaler.fit_transform(data_matrix)

### Data reduction ###
    pca_list = [None]
    if args.pca_set:
        pca_list = [float(f) for f in args.pca_set.split(',')]

    if args.pca_plot:
        pca_set_len = len(pca_list)
        fg, axs = dynamic_subplots(pca_set_len)
        fg.suptitle("PCA variance ratios",fontsize=15)
        for ax, pca_set in zip(axs,pca_list):
            pca = PCA(pca_set)
            pca.fit(data_matrix)

            plot_pca(fg,ax,pca)

            ax.set_title("n_components = {0}".format(pca_set))

        plt.show()
        sys.exit()

    if not args.pca_set:
        pca_list = [0.85,0.9,0.95,0.99]
    else:
        pca_list = [float(f) for f in args.pca_set.split(',')]

### Clustering parameters ###
    if args.dbscan_minpts_plot:
        pca_set_len = len(pca_list)
        fg, axs = dynamic_subplots(pca_set_len)
        fg.suptitle("Mean distances for different n_components",fontsize=15)
        for ax, pca_set in zip(axs,pca_list):
            pca = PCA(pca_set)
            dm_transformed = pca.fit_transform(data_matrix)
            graph = get_sorted_neighbors(dm_transformed,dm_transformed.shape[0]-1)
            means = np.mean(graph,axis=0)
           
            plot_points(fg,ax,means)

            ax.set_title("n_components: set={0}, real={1}".format(pca_set,pca.n_components_))
            ax.set_xlabel("i-th neighbor")
            ax.set_ylabel("mean distance")

        for ax in axs[pca_set_len:]:
            ax.axis('off')

        plt.show()
        sys.exit()

    def pca_minPts_generator():
        dbscan_minPts = []
        for pca in pca_list:
            pca = PCA(pca)
            pca.fit(data_matrix)
            dbscan_minPts.append(pca.n_components_ + 1)
        return zip(pca_list, dbscan_minPts)

    if args.dbscan_eps_plot:
        set_len = len(list(pca_minPts_generator()))
        fg, axs = dynamic_subplots(set_len)
        fg.suptitle(
            "Neighbors distances for different n_components and minPts",
            fontsize=15
        )
        for ax, (pca_set,minPts) in zip(axs, pca_minPts_generator()):
            graph = get_sorted_neighbors(dm_transformed,minPts)
            distances = graph[:,-minPts]
            distances.sort()

            plot_points(fg,ax,distances)

            ax.set_title(
                "{0}-th neighbor distances for n_components={1}".format(
                    minPts, pca_set
                )
            )
            ax.set_xlabel("i-th point")
            ax.set_ylabel("distance")

        for ax in axs[set_len:]:
            ax.axis('off')

        plt.show()
        sys.exit()

    # Solhouette analysis
    if args.silhouette:
        if args.cluster is None:
            print "--cluster required for --silhouette"
            sys.exit()
        clustering = args.cluster

        # Parameters
        if args.silhouette != 'default':
            A = args.silhouette.split(':')
            a,b,n = float(A[0]), float(A[1]), int(A[2])
            params = np.linspace(a,b,n)
        else:
            params = np.linspace(0,0,0)
        # Ellipse
        #    0.85: np.linspace(0.8,1.6,12),
        #    0.9: np.linspace(1.,1.8,12),
        #    0.95: np.linspace(1.3,2.0,12),
        #    0.99: np.linspace(2.0,4.0,12),
        #    0.999: np.linspace(2.3,4.0,12)
        # Circles
        #    0.85: [0.2] + list(np.linspace(0.45,0.6,11)),
        #    0.9: [0.5] + list(np.linspace(0.65,0.7,11)),
        #    0.95: [0.7] + list(np.linspace(0.8,0.9,11)),
        #    0.99: list(np.linspace(1.8,2.0,12)),
        #    0.999: list(np.linspace(2.2,2.5,12)),

        # Figure
        fg, axs = dynamic_subplots(len(list(params_generator())))
        suptitle = {
            'k-means': "KMeans",
            'dbscan': "DBSCAN"
        }[clustering]
        fg.suptitle("{0} silhouette analysis".format(suptitle), fontsize=15)

        # Plot axis
        n_components = float(args.pca_set)

        for ax, param in zip(axs,params):
            # PCA transoform
            pca = PCA(n_components=n_components)
            dm_transformed = pca.fit_transform(data_matrix)

            # TODO
            # Clustering
            Cluster = {
                'k-means': KMeans,
                'dbscan': DBSCAN
            }[clustering]
            cluster = Cluster(**params) 
            cluster.fit(dm_transformed)

            labels = cluster.labels_
            n_clusters = len(np.unique(labels)) + (-1 in labels)

            # Ax title
            if clustering == 'k-means':
                ax_title =  "n_cmps:set={0}|real={1}, clusts:{2}".format(
                    n_components, pca.n_components_, params['n_clusters']
                )
            elif clustering == 'dbscan':
                ax_title = "n_cmps:set={0}|real={1}, eps:{2:.2f}, minPts:{3}, clusts:{4}".format(
                    n_components, pca.n_components_, params['eps'], params['min_samples'],
                    len(np.unique(labels))
                )
            else:
                print cluster
            ax.set_title(ax_title)

            # Plot silhouette values for clusters
            silhouette_avg = metrics.silhouette_score(
                dm_transformed[labels != -1], labels[labels != -1]
            )
            sample_silhouette_values = metrics.silhouette_samples(
                dm_transformed, labels
            )

            y_lower = 10
            for i in xrange(n_clusters):
                ith_cluster_silhouette_values = (
                    sample_silhouette_values[labels == i]
                )

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.Set1(float(i) / n_clusters)
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                    0, ith_cluster_silhouette_values,
                    facecolor=color, edgecolor=color, alpha=0.7
                )
                
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), color=color)

                y_lower = y_upper + 10

            # Outliers Informations for DBSCAN
            if clustering == 'dbscan':
                x_min = ax.get_xlim()[0]
                x_lim = ax.get_xlim()[-1] - ax.get_xlim()[0]
                size_outliers = len(labels[labels == -1])
                color = 'black'
                ax.fill_betweenx(np.arange(y_lower, y_lower+size_outliers),
                    x_min + x_lim * 0.25, x_min + x_lim * 0.75,
                    facecolor=color, edgecolor=color, alpha=0.3
                )
                ax.text( x_min + x_lim * 0.5, y_lower + 0.5 * size_outliers,
                    str(size_outliers), color=color, alpha=0.7,
                    horizontalalignment='center'
                )

            # X axis
            ax.set_xlabel("The silhouette coefficient values")

            # Y axis
            ax.set_ylabel("Cluster label")
            y_max = dm_transformed.shape[0] + (n_clusters + 1) * 10
            ax.set_ylim([0, y_max])
            ax.yaxis.set_major_locator(ticker.LinearLocator(10))

            # Plot silhouette_avg line
            ax.axvline(x=silhouette_avg, color='red', linestyle='--')
            ax.text(
                silhouette_avg + 0.01 * ax.get_xlim()[-1],
                0.90 * y_max, '{0:.3f}'.format(silhouette_avg),
                color='red'
            )

            # Grid
            ax.grid(which='major', alpha=0.7)
            ax.grid(which='minor', alpha=0.3)
   
        fg.subplots_adjust(hspace=0.32)
        plt.show()

        sys.exit()

### Cluster ###
    if args.cluster_analysis:
        if args.cluster is None:
            print "--cluster required for --cluster-analysis"
            sys.exit()
        analysis = args.cluster_analysis

        # DBSCAN
        # Ellipse
        #n_components, minPts, eps = (0.99, 41, 3.09)    #0.419, 604, strawberry
        #n_components, minPts, eps = (0.99, 41, 2.55)   #0.502, 984, grape
        #n_components, minPts, eps = (0.95, 21, 1.6)    #0.511, 917, banana
        #n_components, minPts, eps = (0.9, 14, 1.18)    #0.534, 807, ananas
        # Circle
        #n_components, minPts, eps = (0.85, 6, 0.58)      #0.670
        #n_components, minPts, eps = (0.95, 11, 0.87)    #0.716

        n_components, minPts, eps = (0.95, 1, 100)    #0.716

        # KMeans
        n_clusters = 40

        # PCA transfer
        start = time.clock()
        sys.stdout.write(
            "PCA analysis for n_components={0} ... ".format(n_components)
        )
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(data_matrix)
        sys.stdout.write(
            "{0}s, real n_components={1}\n".format(time.clock()-start,pca.n_components_)
        )

        # Clustering
        start = time.clock()
        if args.cluster == 'dbscan':
            sys.stdout.write(
                "DBSCAN clustering for eps={0}, minPts={1} ... ".format(eps,minPts)
            )
            cluster = DBSCAN(eps=eps, min_samples=minPts)
        elif args.cluster == 'kmeans':
            sys.stdout.write(
                "KMeans clustering for n_clusters={0} ... ".format(n_clusters)
            )
            cluster = KMeans(n_cluster=n_clusters)
        cluster.fit(X)
        labels = cluster.labels_
        uniques = np.unique(labels)
        n_labels = len(uniques)
        n_clusters = n_labels - (-1 in labels)
        sys.stdout.write(
            "{0}s, labels={1}\n".format(time.clock()-start,n_labels)
        )

        # Manifold learning
        start = time.clock()
        if analysis == 'tsne':
            sys.stdout.write(
                "TSNE manifold learning ... "
            )
            manifolder = manifold.TSNE(n_components=2, init='pca', random_state=0)
        elif analysis == 'isomap':
            sys.stdout.write(
                "Isomap manifold learning ... "
            )
            manifolder = manifold.Isomap(n_neighbors=minPts, n_components=2)
        elif analysis == 'mds':
            sys.stdout.write(
                "MDS manifold learning ... "
            )
            manifolder = manifold.MDS(n_components=2)
        X = manifolder.fit_transform(X)
        sys.stdout.write(
            "{0}s\n".format(time.clock() - start)
        )

        # Plot
        col_min, col_max = np.min(X, 0), np.max(X, 0)
        X = (X - col_min) / (col_max - col_min)

        # Figure
        fg, ax = plt.subplots(1,1)
        suptitle = {
            'tsne': "TSNE",
            'isomap': "Isomap",
            'mds': "MDS"
        }[analysis]
        if args.cluster == 'dbscan':
            fg.suptitle(
                "{0} distribution for DBSCAN:n_components={1}, minPts={2}, eps={3}".format(
                    suptitle, n_components, minPts, eps
                ),
                fontsize=15
            )
        elif args.cluster == 'k-means':
            fg.suptitle(
                "{0} distribution for KMeans:n_cluster={1}".format(
                    suptitle, n_clusters
                ),
                fontsize=15
            )
        
        # Color, apha, font styles ...
        colors = {}
        alphas = {}
        font_sizes = {}
        weights = {}
        styles = {}
        for i in xrange(n_clusters):
            colors[i] = cm.Set1(i/float(n_clusters))
            alphas[i] = 0.8
            font_sizes[i] = 13
            weights[i] = 'bold'
            styles[i] = 'normal'
        if -1 in labels:
            colors[-1] = 'black'
            alphas[-1] = 0.2
            font_sizes[-1] = 9 
            weights[-1] = 'light'
            styles[-1] = 'italic'

        # Dir names (for save)
        if args.out_dir is not None:
            dirs = {}
            os.mkdir(args.out_dir, 0755)
            for i in uniques:
                dir_name = "{0}/cluster_{1}".format(args.out_dir,i)
                os.mkdir(dir_name, 0755)
                dirs[i] = dir_name + '/'
  
        # Scatter samples as letters
        start = time.clock()
        last = start
        sys.stdout.write("Scattering {0} ...\n".format(X.shape[0]))

        file_names = [os.path.split(f)[-1] for f in files]
        for i in xrange(X.shape[0]):
            label = labels[i]
            plt.text(X[i,0],X[i,1],
                file_names[i].split('_')[1][0],
                color=colors[label], alpha=alphas[label],
                fontdict={'weight':'bold', 'size':font_sizes[label]}
            )

            # Save
            if args.out_dir is not None:
                Z, extent = get_image_grid(data_list[i])
                tfig, tax = plt.subplots(1,1)
                tim = tax.imshow(Z, extent=extent, cmap=cm.coolwarm, aspect='equal',
                    origin='lower')
                tfig.colorbar(tim,shrink=0.8)
                plt.savefig(
                    dirs[label] + str(i) + '_' +  os.path.splitext(file_names[i])[0] + '.png'
                )
                plt.close(tfig)

                if time.clock() - last > 2:
                    sys.stdout.write("... {0} ploted and saved\n".format(i))
                    last = time.clock()

        sys.stdout.write("... {0}s\n".format(time.clock()-start))

        # Add AnnotationBbox
        if hasattr(offsetbox, 'AnnotationBbox'):
            sys.stdout.write("Adding AnnotationBboxes ...\n")

            shown_images = np.array([[1.,1.]])
            for i in xrange(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    get_offset_image(data_list[i]),
                    X[i],
                    frameon=False
                )
                ax.add_artist(imagebox)
                label = labels[i]
                anotation = file_names[i].split('_')[1][:3]+str(i)
                if label == -1:
                    anotation = '(' + anotation + ')'
                else:
                    anotation=str(label) + anotation
                ax.text(X[i,0], X[i,1] + 0.03,
                    anotation,
                    color=colors[label],
                    fontdict={'weight':weights[label], 'size':13},
                    horizontalalignment='center',
                    alpha=alphas[label]+0.2,
                    style=styles[label]
                )

        # Add legend
        class AnyObjectHandler(object):
            def legend_artist(self, legend, i, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                patch = mtext.Text(
                    text='abc', color=colors[i], alpha=alphas[i],
                    style=styles[i], fontsize=font_sizes[i], weight=weights[i],
                )
                handlebox.add_artist(patch)
                return patch

        objs = []
        lables = []
        handlers = []
        for i in uniques:
            objs.append(i)
            lables.append("cluster {0}".format(i))
            handlers.append(AnyObjectHandler())
        ax.legend(
            objs, lables, handler_map=dict(zip(objs,handlers)),
            bbox_to_anchor=(1.01,1), loc='upper left',
            borderaxespad=0.
        )
            
        # Axes
        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()

        sys.exit()
