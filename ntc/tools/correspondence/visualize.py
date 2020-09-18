import matplotlib.pyplot as plt
import numpy as np
import npy
import ntc
from npy import task, Rowof, Colof

__all__ = ['scaled_scatter',
           'plot_matchvolume_and_confidence',
           'plot_matchvolume',
           'plot_matchvolume_and_anomalymap'
           ]


def scaled_scatter(ax, wi, hi, c, sz, L, S, H, W):
    wi = wi.astype(np.float32) * (W - S) // (L - 1) + S // 2  # scale
    hi = hi.astype(np.float32) * (H - S) // (L - 1) + S // 2  # scale
    ax.scatter(x=wi.reshape([-1]), y=hi.reshape([-1]), c=c.reshape([-1, 4]), s=sz.reshape([-1]))


def plot_matchvolume_and_confidence(img0, img1, L, matchvolume_AB, matchvolume_A, matchvolume_B, A_B, B_A, sz_max=2000):
    H, W = img0.shape[:2]
    S = 256 / L

    fig, axes = plt.subplots(ncols=3, nrows=3)
    fig.set_size_inches(2.5 * 3, 2.5 * 3)

    with Rowof(axes, 0) as row:
        with Colof(row, 0) as ax:
            npy.image.show(img0, ax=ax)
            ax.set_title('Image (src)')

        with Colof(row, 1) as ax:
            npy.image.show(img1, ax=ax)
            ax.set_title('Image (tgt)')

        with Colof(row, 2) as ax:
            ax.set_axis_off()
            ax.set_title('Match map')
            h0, w0 = np.meshgrid(np.linspace(-1, 1, L), np.linspace(-1, 1, L))
            hi0, wi0 = np.meshgrid(np.arange(0, L), np.arange(0, L))
            colorvalues = np.arctan2(h0, w0) / 2 / np.pi + 0.5
            colors = plt.cm.rainbow(colorvalues)
            s0 = np.maximum(abs(h0 * 2 / L), abs(w0 * 2 / L)) ** 2 * sz_max

            c0 = colors[hi0, wi0]

            ax.imshow(np.full((256, 256), 255, dtype=np.uint8), cmap='gray_r', vmax=255, vmin=0)
            scaled_scatter(ax, wi0, hi0, c0, s0, L, S, H, W)

    with Rowof(axes, 1) as row:
        with Colof(row, 0) as ax:
            ax.set_title('NN (src)')
            hi1, wi1 = np.meshgrid(np.arange(0, L), np.arange(0, L))
            maps = matchvolume_A[hi1, wi1]  # corres_t(h1, w1) is the corresponding h0, w0

            m1 = np.all(maps >= 0, axis=-1)
            hi1 = hi1[m1]
            wi1 = wi1[m1]
            h1_0, w1_0 = maps[m1].T
            c1 = colors[h1_0, w1_0]
            s1 = s0[h1_0, w1_0]

            npy.image.show(img0, ax=ax)
            scaled_scatter(ax, wi1, hi1, c1, s1, L, S, H, W)

        with Colof(row, 1) as ax:
            ax.set_title('NN (tgt)')
            hi1, wi1 = np.meshgrid(np.arange(0, L), np.arange(0, L))
            maps = matchvolume_B[hi1, wi1]  # corres_t(h1, w1) is the corresponding h0, w0

            m1 = np.all(maps >= 0, axis=-1)
            hi1 = hi1[m1]
            wi1 = wi1[m1]
            h1_0, w1_0 = maps[m1].T
            c1 = colors[h1_0, w1_0]
            s1 = s0[h1_0, w1_0]

            npy.image.show(img1, ax=ax)
            scaled_scatter(ax, wi1, hi1, c1, s1, L, S, H, W)

        with Colof(row, 2) as ax:
            ax.set_title('Match (mutual NN)')
            hi1, wi1 = np.meshgrid(np.arange(0, L), np.arange(0, L))
            maps = matchvolume_AB[hi1, wi1]  # corres_t(h1, w1) is the corresponding h0, w0

            m1 = np.all(maps >= 0, axis=-1)
            hi1 = hi1[m1]
            wi1 = wi1[m1]
            h1_0, w1_0 = maps[m1].T
            c1 = colors[h1_0, w1_0]
            s1 = s0[h1_0, w1_0]

            npy.image.show(img1, ax=ax)
            scaled_scatter(ax, wi1, hi1, c1, s1, L, S, H, W)

    with Rowof(axes, 2) as row:
        import seaborn as sns
        def plot_confidence(ax_, data):
            sns.heatmap(data, vmin=0, vmax=100, cmap='gray', annot=True, fmt='.0f', ax=ax_, cbar=False,
                        annot_kws={'fontsize': 5})

        with Colof(row, 0) as ax:
            ax.set_axis_off()
            A_B = A_B.detach().cpu().numpy().max(axis=0)
            plot_confidence(ax, A_B * 100)
            ax.set_title('Confidence (src)')

        with Colof(row, 1) as ax:
            ax.set_axis_off()
            B_A = B_A.detach().cpu().numpy().max(axis=0)
            plot_confidence(ax, B_A * 100)
            ax.set_title('Confidence (tgt)')

        with Colof(row, 2) as ax:
            npy.image.blank(ax)

    plt.tight_layout()


def plot_matchvolume(img0, img1, matchvolume, sz_max=2000):
    H, W = img0.shape[:2]
    L = matchvolume.shape[-2]
    S = 256 // L

    fig, axes = plt.subplots(ncols=2, nrows=2)
    fig.set_size_inches(2.5 * 2, 2.5 * 2)

    with Rowof(axes, 0) as row:
        with Colof(row, 0) as ax:
            npy.image.show(img0, ax=ax)
            ax.set_title('Image (src)')

        with Colof(row, 1) as ax:
            npy.image.show(img1, ax=ax)
            ax.set_title('Image (tgt)')

    with Rowof(axes, 1) as row:
        with Colof(row, 0) as ax:
            ax.set_title('Match map')
            h0, w0 = np.meshgrid(np.linspace(-1, 1, L), np.linspace(-1, 1, L))
            hi0, wi0 = np.meshgrid(np.arange(0, L), np.arange(0, L))
            colorvalues = np.arctan2(h0, w0) / 2 / np.pi + 0.5
            colors = plt.cm.rainbow(colorvalues)
            # s0 = np.maximum(abs(h0), abs(w0)) * sz_max
            s0 = np.maximum(abs(h0 * 2 / L), abs(w0 * 2 / L)) ** 2 * sz_max

            c0 = colors[hi0, wi0]

            npy.image.show(img0, ax=ax)
            scaled_scatter(ax, wi0, hi0, c0, s0, L, S, H, W)

        with Colof(row, 1) as ax:
            ax.set_title('Nearest Neighbor')
            hi1, wi1 = np.meshgrid(np.arange(0, L), np.arange(0, L))
            maps = matchvolume[hi1, wi1]  # corres_t(h1, w1) is the corresponding h0, w0

            m1 = np.all(maps >= 0, axis=-1)
            hi1 = hi1[m1]
            wi1 = wi1[m1]
            h1_0, w1_0 = maps[m1].T
            c1 = colors[h1_0, w1_0]
            s1 = s0[h1_0, w1_0]

            npy.image.show(img1, ax=ax)
            scaled_scatter(ax, wi1, hi1, c1, s1, L, S, H, W)

    plt.tight_layout()


def plot_matchvolume_and_anomalymap(img0, img1, T, matchvolume, anomaly_map, anomaly_score):
    H, W = img0.shape[:2]
    L = matchvolume.shape[-2]
    S = 256 / L
    sz_max = 24

    fig, axes = plt.subplots(ncols=2, nrows=4)
    fig.set_size_inches(2.5 * 2, 2.5 * 4)

    with Rowof(axes, 0) as row:
        with Colof(row, 0) as ax:
            npy.image.show(img0, ax=ax)
            ax.set_title('Image (src)')

        with Colof(row, 1) as ax:
            npy.image.show(img1, ax=ax)
            ax.set_title('Image (tgt)')

    with Rowof(axes, 1) as row:
        with Colof(row, 0) as ax:
            ax.set_title('Correspondence (src)')
            h0, w0 = np.meshgrid(np.linspace(-1, 1, L), np.linspace(-1, 1, L))
            hi0, wi0 = np.meshgrid(np.arange(0, L), np.arange(0, L))
            colorvalues = np.arctan2(h0, w0) / 2 / np.pi + 0.5
            colors = plt.cm.rainbow(colorvalues)
            s0 = np.maximum(abs(h0), abs(w0)) * sz_max

            c0 = colors[hi0, wi0]

            npy.image.show(img0, ax=ax)
            scaled_scatter(ax, wi0, hi0, c0, s0, L, S, H, W)

        with Colof(row, 1) as ax:
            ax.set_title('Correspondence (tgt)')
            hi1, wi1 = np.meshgrid(np.arange(0, L), np.arange(0, L))
            maps = matchvolume[hi1, wi1]  # corres_t(h1, w1) is the corresponding h0, w0

            m1 = np.all(maps >= 0, axis=-1)
            hi1 = hi1[m1]
            wi1 = wi1[m1]
            h1_0, w1_0 = maps[m1].T
            c1 = colors[h1_0, w1_0]
            s1 = s0[h1_0, w1_0]

            npy.image.show(img1, ax=ax)
            scaled_scatter(ax, wi1, hi1, c1, s1, L, S, H, W)

    with Rowof(axes, 2) as row:
        T = ntc.to_numpy(T)
        marg_src = np.sum(T, axis=(2, 3))
        marg_tgt = np.sum(T, axis=(0, 1))
        # print(f'Sum: | T | {T.sum():.2f} | Marg (src) | {marg_src.sum():.2f} | Marg (tgt) | {marg_tgt.sum():.2f}')

        with Colof(row, 0) as ax:
            ax.imshow(marg_src, cmap='gray', vmax=marg_src.max(), vmin=0)
            ax.set_title(f'Marginal (src, sum: {marg_src.sum():.2f})')
            set_axis_line(ax)

        with Colof(row, 1) as ax:
            ax.imshow(marg_tgt, cmap='gray', vmax=marg_tgt.max(), vmin=0)
            ax.set_title(f'Marginal (tgt, sum: {marg_tgt.sum():.2f})')
            set_axis_line(ax)

    with Rowof(axes, 3) as row:
        with Colof(row, 0) as ax:
            npy.image.blank(ax)
            ax.set_title(f'Anomaly score: {anomaly_score:.4f}')

        with Colof(row, 1) as ax:
            ax.imshow(anomaly_map, cmap='gray', vmax=max(anomaly_map.max(), 1e-2))
            set_axis_line(ax)
            ax.set_title('Anomaly map')

    plt.tight_layout()


def set_axis_line(ax):
    ax.set_yticks([])
    ax.set_xticks([])
