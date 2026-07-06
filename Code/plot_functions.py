import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
params = {'mathtext.default': 'regular',
          'figure.dpi': 600}      
plt.rcParams.update(params)

from archetypes import AA


def _archetype_prefix(archetype_prefix: str = 'A') -> str:
    prefix = 'A' if archetype_prefix is None else str(archetype_prefix).strip()
    return prefix or 'A'


def _archetype_labels(n_archetypes: int, archetype_prefix: str = 'A') -> list[str]:
    prefix = _archetype_prefix(archetype_prefix)
    return [f'{prefix}{i+1}' for i in range(n_archetypes)]


def _archetype_labels_long_default(n_archetypes: int, archetype_prefix: str = 'A') -> list[str]:
    prefix = _archetype_prefix(archetype_prefix)
    if prefix == 'A':
        return [f'Archetype {i+1}' for i in range(n_archetypes)]
    return _archetype_labels(n_archetypes, prefix)


def plot_archetypes(X, 
                    n_archetypes: int = 3,
                    tolerance: float = 1e-4,
                    max_iter: int = 1000,
                    n_init: int = 5,
                    init: str = 'furthest_sum',
                    random_state: int = 42):
    """
    Calculate archetypes and plot in PCA space.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
        Data matrix.
    n_archetypes : int, default=3
        Number of archetypes to compute.
    tolerance : float, default=1e-4
        Convergence tolerance for AA.
    max_iter : int, default=1000
        Maximum number of iterations for AA.
    n_init : int, default=5
        Number of initializations for AA.
    init : str, default='furthest_sum'
        Initialization method for AA.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    aa : AA
        Fitted archetypes model.
    """

    # Keep index information for label alignment
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)

    # Fit AA
    aa = AA(n_archetypes=n_archetypes, random_state=random_state, 
            tol=tolerance, max_iter=max_iter, n_init=n_init, init=init)
    aa.fit(X_df.values)

    return aa


def plot_simplex(aa, meta,
                 include_class: bool = True,
                 levels: int = 5,
                 tick_size: int = 12,
                 archetype_prefix: str = 'A'):
    """
    Plot samples in a 2-D simplex (equilateral triangle) for *three* archetypes.

    Parameters
    ----------
    aa : fitted AA object
        Must have attribute ``coefficients_`` with shape (n_samples, 3).
    meta : pandas.DataFrame
        Metadata table. If it contains a ``'class'`` column, points are
        colored by that column.
    include_class : bool, default=True
        Whether to color points by class.
    levels : int, default=5
        Number of grid subdivisions (e.g. 5 → 20 %).
    tick_size : int, default=12
        Font size of edge tick labels.
    archetype_prefix : str, default='A'
        Prefix used for archetype labels on the plot (e.g. 'W' or 'U').
    """

    coefficients = aa.coefficients_

    # vertices of an equilateral triangle (simplex)
    v0 = np.array([0.0, 0.0])                    # A1 = 100 %
    v1 = np.array([1.0, 0.0])                    # A2 = 100 %
    v2 = np.array([0.5, np.sqrt(3) / 2])         # A3 = 100 %

    # helper: barycentric → Cartesian
    def bary(a1, a2, a3):
        return a1 * v0 + a2 * v1 + a3 * v2

    # convert data points
    coords = coefficients @ np.vstack([v0, v1, v2])
    x, y = coords[:, 0], coords[:, 1]

    # plot
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    cmap = ListedColormap(['tomato',
                           'steelblue', 
                           'olivedrab', 
                           'darkorange',
                           'brown', 
                           'dodgerblue', 
                           'purple', 
                           'navy'])

    # scatter
    if include_class and "class" in meta.columns:
        classes = meta["class"].astype(str).values
        uniq = np.unique(classes)
        for i, name in enumerate(uniq):
            idx = classes == name
            ax.scatter(
                x[idx], y[idx],
                s=50, alpha=0.5,
                label=name, 
                color=cmap(i)
            )
    else:
        ax.scatter(x, y, s=40, alpha=0.5, color="gray")

    # simplex outline
    for p, q in [(v0, v1), (v1, v2), (v2, v0)]:
        ax.plot([p[0], q[0]], [p[1], q[1]], lw=1.2, color="black")

    # interior grid
    for i in range(1, levels):
        f = i / levels
        # constant A1
        ax.plot(*zip(*[bary(f, 1 - f, 0), bary(f, 0, 1 - f)]),
                lw=0.5, color="gray", alpha=0.4)
        # constant A2
        ax.plot(*zip(*[bary(0, f, 1 - f), bary(1 - f, f, 0)]),
                lw=0.5, color="gray", alpha=0.4)
        # constant A3
        ax.plot(*zip(*[bary(1 - f, 0, f), bary(0, 1 - f, f)]),
                lw=0.5, color="gray", alpha=0.4)

    fmt = lambda k: f"{int(100 * k / levels)}%"

    # edge tick labels
    # A1 axis – bottom edge v0-v1 (rightward increasing A1)
    for i in range(levels + 1):
        p = bary(i / levels, 1 - i / levels, 0)          # A3 = 0
        ax.text(p[0], p[1] - 0.035, fmt(i),
                ha="center", va="top", fontsize=tick_size)

    # A2 axis – right edge v1-v2 (upward decreasing A2)
    for i in range(levels + 1):
        p = bary(0, 1 - i / levels, i / levels)          # A1 = 0
        ax.text(p[0] + 0.035, p[1], fmt(levels - i),
                ha="left", va="center", fontsize=tick_size)

    # A3 axis – left edge v0-v2 (upward increasing A3)
    for i in range(levels + 1):
        p = bary(1 - i / levels, 0, i / levels)          # A2 = 0
        ax.text(p[0] - 0.035, p[1], fmt(i),
                ha="right", va="center", fontsize=tick_size)

    # vertex labels
    labels = _archetype_labels(3, archetype_prefix)
    ax.text(v0[0] - 0.15, v0[1] - 0.08, labels[0], fontsize=15, fontweight='bold')
    ax.text(v1[0] + 0.05, v1[1] - 0.08, labels[1], fontsize=15, fontweight='bold')
    ax.text(v2[0], v2[1] + 0.05, labels[2], ha="center", fontsize=15, fontweight='bold')

    # formatting
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, v2[1] + 0.08)
    ax.axis("off")

    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.05), fontsize=11, frameon=False)
    plt.tight_layout()
    plt.show()


def plot_simplex_by_reactors(aa, meta,
                             include_class: bool = True,
                             levels: int = 5,
                             tick_size: int = 12,
                             archetype_prefix: str = 'A'):
    """
    Plot samples in a 2-D simplex (equilateral triangle) for *three* archetypes,
    using **different markers** for groups derived from the *prefix* of the `meta`
    index (substring before the first underscore). Color encodes class labels
    (when available via `meta['class']`), while marker encodes the prefix group.

    Parameters
    ----------
    aa : fitted AA object
        Must have attribute ``coefficients_`` with shape (n_samples, 3).
    meta : pandas.DataFrame
        Metadata table. If it contains a ``'class'`` column, points are
        colored by that column. The **index** is used to derive group labels.
    include_class : bool, default=True
        Whether to color points by class (when `meta['class']` is present).
    levels : int, default=5
        Number of grid subdivisions (e.g. 5 → 20 %).
    tick_size : int, default=12
        Font size of edge tick labels.
    archetype_prefix : str, default='A'
        Prefix used for archetype labels on the plot (e.g. 'W' or 'U').
    """

    coefficients = aa.coefficients_
    n = coefficients.shape[0]

    # vertices of an equilateral triangle (simplex)
    v0 = np.array([0.0, 0.0])                    # A1 = 100 %
    v1 = np.array([1.0, 0.0])                    # A2 = 100 %
    v2 = np.array([0.5, np.sqrt(3) / 2])         # A3 = 100 %

    # helper: barycentric → Cartesian
    def bary(a1, a2, a3):
        return a1 * v0 + a2 * v1 + a3 * v2

    # convert data points
    coords = coefficients @ np.vstack([v0, v1, v2])
    x, y = coords[:, 0], coords[:, 1]

    # ------ Build class labels (color) ------
    classes = None
    class_names = np.array([])
    classes_arr = None
    if include_class and isinstance(meta, pd.DataFrame) and ('class' in meta.columns):
        if len(meta) == n:
            classes = meta['class'].copy().reset_index(drop=True)
            labeled_mask = classes.notna() & (classes.astype(str).str.strip() != '')
            class_names = np.unique(classes[labeled_mask].astype(str).values)
            classes_arr = classes.astype(str).values
        else:
            # length mismatch → treat as unlabeled
            classes = None

    # ------ Build marker groups from index prefix (group) ------
    if isinstance(meta, pd.DataFrame) and (len(meta) == n):
        idx_for_groups = meta.index
    else:
        idx_for_groups = pd.RangeIndex(n)

    groups_series = pd.Index(idx_for_groups).astype(str).to_series(index=np.arange(n))
    groups_series = groups_series.str.split("_", n=1, expand=False).str[0]
    groups_series = groups_series.fillna("Unknown").astype(str)
    groups_arr = groups_series.values
    group_names = np.unique(groups_arr)

    # Marker palette
    marker_cycle = [
        "o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "H", "d", "p"
    ]
    group_to_marker = {
        name: marker_cycle[i % len(marker_cycle)] for i, name in enumerate(group_names)
    }

    # plot
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    cmap = ListedColormap(['tomato',
                           'steelblue',
                           'olivedrab',
                           'darkorange',
                           'brown',
                           'dodgerblue',
                           'purple',
                           'navy'])

    # scatter by (class color) × (group marker)
    if include_class and classes_arr is not None and class_names.size:
        for i, cname in enumerate(class_names):
            for g in group_names:
                mask = (classes_arr == cname) & (groups_arr == g)
                if np.any(mask):
                    ax.scatter(
                        x[mask], y[mask],
                        s=50, alpha=0.5,
                        color=cmap(i),
                        marker=group_to_marker[g],
                    )
    else:
        # No classes: use grey color but different markers per group
        for g in group_names:
            mask = (groups_arr == g)
            if np.any(mask):
                ax.scatter(
                    x[mask], y[mask],
                    s=50, alpha=0.5,
                    color="gray",
                    marker=group_to_marker[g],
                )

    # simplex outline
    for p, q in [(v0, v1), (v1, v2), (v2, v0)]:
        ax.plot([p[0], q[0]], [p[1], q[1]], lw=1.2, color="black")

    # interior grid
    for i in range(1, levels):
        f = i / levels
        # constant A1
        ax.plot(*zip(*[bary(f, 1 - f, 0), bary(f, 0, 1 - f)]),
                lw=0.5, color="gray", alpha=0.4)
        # constant A2
        ax.plot(*zip(*[bary(0, f, 1 - f), bary(1 - f, f, 0)]),
                lw=0.5, color="gray", alpha=0.4)
        # constant A3
        ax.plot(*zip(*[bary(1 - f, 0, f), bary(0, 1 - f, f)]),
                lw=0.5, color="gray", alpha=0.4)

    fmt = lambda k: f"{int(100 * k / levels)}%"

    # edge tick labels
    # A1 axis – bottom edge v0-v1 (rightward increasing A1)
    for i in range(levels + 1):
        p = bary(i / levels, 1 - i / levels, 0)          # A3 = 0
        ax.text(p[0], p[1] - 0.035, fmt(i),
                ha="center", va="top", fontsize=tick_size)

    # A2 axis – right edge v1-v2 (upward decreasing A2)
    for i in range(levels + 1):
        p = bary(0, 1 - i / levels, i / levels)          # A1 = 0
        ax.text(p[0] + 0.035, p[1], fmt(levels - i),
                ha="left", va="center", fontsize=tick_size)

    # A3 axis – left edge v0-v2 (upward increasing A3)
    for i in range(levels + 1):
        p = bary(1 - i / levels, 0, i / levels)          # A2 = 0
        ax.text(p[0] - 0.035, p[1], fmt(i),
                ha="right", va="center", fontsize=tick_size)

    # vertex labels
    labels = _archetype_labels(3, archetype_prefix)
    ax.text(v0[0] - 0.15, v0[1] - 0.08, labels[0], fontsize=15, fontweight='bold')
    ax.text(v1[0] + 0.05, v1[1] - 0.08, labels[1], fontsize=15, fontweight='bold')
    ax.text(v2[0], v2[1] + 0.05, labels[2], ha="center", fontsize=15, fontweight='bold')

    # formatting
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, v2[1] + 0.08)
    ax.axis("off")

    # Legend: classes by color, groups by marker
    handles = []
    if include_class and classes_arr is not None and class_names.size:
        class_handles = [
            Patch(
                facecolor=cmap(i),
                edgecolor=cmap(i),
                alpha=0.5,
                label=str(cname),
            )
            for i, cname in enumerate(class_names)
        ]
        handles.extend(class_handles)

    group_handles = [
        Line2D(
            [0], [0],
            marker=group_to_marker[g],
            color="grey",
            markerfacecolor="grey",
            markeredgecolor="k",
            linestyle="None",
            markersize=7,
            label=str(g),
        )
        for g in group_names
    ]
    handles.extend(group_handles)

    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.15, 1.05),
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()
    plt.show()


def plot_stack_bar(aa, meta, continuous_time: bool = False, date_col: str | None = None,
                   archetype_prefix: str = 'A'):
    """
    Plot stacked bar chart of archetype compositions over time.

    Parameters
    ----------
    aa : fitted AA object
        Must have attribute ``coefficients_`` with shape (n_samples, n_archetypes).
    meta : pandas.DataFrame
        Metadata table. Must contain columns ``'class'`` and ``'EF'``.
    continuous_time : bool, default=False
        If False, use a discrete/categorical x-axis (one equally spaced bar per sample).
        If True, use a continuous datetime x-axis so gaps between samples reflect the
        real time differences.
    date_col : str | None, default=None
        Optional column in ``meta`` to use as the time variable when
        ``continuous_time=True``. If None, ``meta.index`` is used.
    archetype_prefix : str, default='A'
        Prefix used for archetype labels on the plot (e.g. 'W' or 'U').
    """

    # Build dataframe
    a_cols = [f'A{k+1}' for k in range(aa.n_archetypes)]
    a_labels = _archetype_labels(aa.n_archetypes, archetype_prefix)
    archetype_label_map = dict(zip(a_cols, a_labels))
    alpha_df = pd.DataFrame(aa.coefficients_, columns=a_cols)
    alpha_df['EF_class'] = meta['class'].astype(str).values
    alpha_df['EF'] = pd.to_numeric(meta['EF'], errors='coerce').values

    if date_col is not None:
        time_raw = meta[date_col]
    else:
        time_raw = meta.index

    time_dt = pd.to_datetime(time_raw, errors='coerce')
    alpha_df['Date_dt'] = time_dt
    alpha_df['Date'] = pd.Index(time_raw).astype(str)

    # Bars
    fig, ax = plt.subplots(figsize=(10, 4))
    palette = ['olivedrab',
               'darkorange',
               'brown',
               'dodgerblue',
               'purple',
               'navy']
    colors = palette[:len(a_cols)]

    if continuous_time and alpha_df['Date_dt'].notna().all():
        alpha_df = alpha_df.sort_values('Date_dt').reset_index(drop=True)
        x_dt = pd.to_datetime(alpha_df['Date_dt'])
        x_num = mdates.date2num(x_dt)

        if len(x_num) == 1:
            bar_width = 1.0
        else:
            diffs = np.diff(x_num)
            bar_width = max(np.min(diffs) * 0.8, 0.2)

        bottom = np.zeros(len(alpha_df), dtype=float)
        for col_name, color in zip(a_cols, colors):
            vals = pd.to_numeric(alpha_df[col_name], errors='coerce').fillna(0).to_numpy(dtype=float)
            ax.bar(x_dt, vals, bottom=bottom, width=bar_width, color=color, align='center',
                   label=archetype_label_map[col_name])
            bottom = bottom + vals

        ax.set_ylabel('Archetype Composition')
        ax.xaxis_date()
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Line on secondary y
        ax2 = ax.twinx()
        ax2.plot(x_dt, alpha_df['EF'], marker='o', color='k', markersize=3, linewidth=1,
                 label='EF', zorder=5)
        ax2.set_ylabel('$N_2O$ Emission Factor (-)')
        ax2.set_xlim(ax.get_xlim())

        # Shade High emission blocks
        high_mask = alpha_df['EF_class'].eq('High emission').to_numpy()
        idx = np.flatnonzero(high_mask)
        if idx.size:
            splits = np.where(np.diff(idx) > 1)[0] + 1
            for g in np.split(idx, splits):
                start, end = int(g[0]), int(g[-1])
                ax.axvspan(x_num[start] - bar_width / 2,
                            x_num[end] + bar_width / 2,
                            facecolor='k', alpha=0.3, zorder=0)
    else:
        if continuous_time and alpha_df['Date_dt'].isna().any():
            warnings.warn(
                "Could not parse all dates for continuous_time=True; falling back to discrete x-axis.",
                UserWarning
            )

        alpha_df['Date'] = alpha_df['Date_dt'].dt.strftime('%Y-%m-%d').fillna(alpha_df['Date'])
        alpha_df.set_index('Date')[a_cols].rename(columns=archetype_label_map).plot(
            kind='bar', stacked=True, ax=ax, color=colors)
        ax.set_ylabel('Archetype Composition')

        # Line on secondary y
        ax2 = ax.twinx()
        x = np.arange(len(alpha_df))  # bar centers
        ax2.plot(x, alpha_df['EF'], marker='o', color='k', markersize=3, linewidth=1,
                 label='EF', zorder=5)
        ax2.set_ylabel('$N_2O$ Emission Factor (-)')
        ax2.set_xlim(ax.get_xlim())

        # Shade High emission blocks
        high_mask = alpha_df['EF_class'].eq('High emission').to_numpy()
        idx = np.flatnonzero(high_mask)
        if idx.size:
            splits = np.where(np.diff(idx) > 1)[0] + 1
            for g in np.split(idx, splits):
                start, end = int(g[0]), int(g[-1])
                ax.axvspan(start - 0.5, end + 0.5, facecolor='k', alpha=0.3, zorder=0)

    # Legend outside (top-right), combining bars + line
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2,
              loc='upper left', bbox_to_anchor=(1.08, 1.0),
              frameon=False)

    # X-ticks
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    # plt.setp(ax.get_yticklabels(), fontsize=13)
    # plt.setp(ax2.get_yticklabels(), fontsize=13)
    


    fig.tight_layout()
    plt.show()

    return alpha_df


def plot_stack_bar_by_reactors(aa, meta: pd.DataFrame,
                               sample_index: pd.Index | None = None,
                               emission_col: str = 'EF',
                               class_col: str = 'class',
                               high_label: str = 'High emission',
                               meta_key: str | None = None,
                               ncols: int = 2,
                               date_fmt: str = '%Y-%m-%d',
                               fontsize: int = 15,
                               archetype_prefix: str = 'A'):
    """
    Plot stacked bar charts of archetype compositions per group (prefix of `meta` index),
    with the EF time series overlaid. One subplot per group.

    Group = substring before first underscore in the index (e.g., 'SBR1' from 'SBR1_2021-01-01').
    Time axis = substring after the first underscore parsed as datetime when possible; otherwise
    falls back to parsing the full index or using string order.

    Parameters
    ----------
    aa : fitted AA object
        Must have attribute ``coefficients_`` with shape (n_samples, n_archetypes).
    meta : pandas.DataFrame
        Metadata table. Expected to have index like "xxx_xxxxxxxxxx".
    sample_index : pandas.Index | sequence | None, optional
        If provided, used to align `meta` to the samples used to fit `aa`.
        This should be the index of the X that went into AA.
    emission_col : str, default 'EF'
        Column name in `meta` for the continuous emission variable.
    class_col : str, default 'class'
        Column name in `meta` for the class labels (used for shading when it equals `high_label`).
    high_label : str, default 'High emission'
        Label in `class_col` indicating high emission samples to be shaded.
    meta_key : str | None, default None
        If provided, the name of the column in `meta` to use as index/key for alignment with `sample_index`.
    ncols : int, default 2
        Number of subplot columns.
    date_fmt : str, default '%Y-%m-%d'
        Date format for x-tick labels when a datetime can be parsed.
    fontsize : int, default 15
        Font size for axis labels and ticks.
    archetype_prefix : str, default='A'
        Prefix used for archetype labels on the plot (e.g. 'W' or 'U').
    """

    # Archetype weights
    k = int(getattr(aa, 'n_archetypes', getattr(aa, 'n_components', None) or aa.coefficients_.shape[1]))
    a_cols = [f'A{i+1}' for i in range(k)]
    a_labels = _archetype_labels(k, archetype_prefix)
    archetype_label_map = dict(zip(a_cols, a_labels))
    alpha_df = pd.DataFrame(aa.coefficients_, columns=a_cols)
    n = len(alpha_df)

    # Robust alignment of `meta`
    if not isinstance(meta, pd.DataFrame):
        m = pd.DataFrame(index=range(n))
        meta_index_for_groups = pd.Index(range(n))
    else:
        meta_local = meta.copy()
        if meta_key is not None and meta_key in meta_local.columns:
            meta_local = meta_local.set_index(meta_key)
        if not meta_local.index.is_unique:
            meta_local = meta_local[~meta_local.index.duplicated(keep='first')]

        if sample_index is not None:
            idx = pd.Index(sample_index)
            m = meta_local.reindex(idx)
            if (m.shape[0] == len(idx)) and (m.isna().all().all()):
                m = (meta_local.rename_axis('key').reset_index()
                               .assign(key=lambda d: d['key'].astype(str))
                               .set_index('key')
                               .reindex(idx.astype(str)))
                m.index = idx
            meta_index_for_groups = idx
        elif len(meta_local) == n:
            m = meta_local.reset_index(drop=True)
            meta_index_for_groups = meta_local.index
        else:
            m = pd.DataFrame(index=range(n))
            meta_index_for_groups = pd.Index(range(n))

    # Derive prefix (group) and suffix (time token) from index
    def _split_index(idx):
        s = pd.Index(idx).astype(str).to_series(index=idx)
        parts = s.str.split('_', n=1, expand=True)
        if isinstance(parts, pd.DataFrame) and parts.shape[1] == 2:
            prefix = parts.iloc[:, 0]
            suffix = parts.iloc[:, 1]
        else:
            prefix = s
            suffix = pd.Series([None]*len(s), index=s.index)
        return prefix, suffix

    prefix_ser, suffix_ser = _split_index(meta_index_for_groups)
    # Align to n rows
    if len(prefix_ser) != n:
        prefix_ser = pd.Series(['Unknown'] * n)
        suffix_ser = pd.Series([None] * n)
    groups = prefix_ser.reset_index(drop=True).fillna('Unknown').astype(str)

    # Build dates/labels and sort keys
    # Try parsing suffix as datetime first
    dt_from_suffix = pd.to_datetime(suffix_ser.reset_index(drop=True), errors='coerce')
    if dt_from_suffix.notna().any():
        order_key = dt_from_suffix
        date_labels = dt_from_suffix.dt.strftime(date_fmt).fillna(suffix_ser.astype(str)).to_list()
    else:
        # Fall back to parsing the whole label or positional order
        if sample_index is not None:
            idx_try = pd.Index(sample_index)
        elif isinstance(meta, pd.DataFrame) and len(meta) == n:
            idx_try = pd.Index(meta.index)
        else:
            idx_try = pd.Index(range(n))
        dt = pd.to_datetime(idx_try, errors='coerce')
        if dt.notna().any():
            order_key = dt
            date_labels = dt.strftime(date_fmt).to_list()
        else:
            order_key = pd.RangeIndex(start=0, stop=n)
            date_labels = idx_try.astype(str).to_list()

    # Combine working frame
    df = alpha_df.copy()
    df['EF'] = pd.to_numeric(m.get(emission_col, pd.Series([np.nan]*n, index=df.index)), errors='coerce').values
    df['EF_class'] = m.get(class_col, pd.Series([pd.NA]*n, index=df.index)).astype('string').values
    df['Date'] = date_labels
    df['__order__'] = order_key.values
    df['Group'] = groups.values

    # Global y-limits
    # Primary axis (stacked composition)
    primary_ylim = (0.0, 1.0)

    # Secondary axis (emission)
    ef_all = pd.to_numeric(df['EF'], errors='coerce').to_numpy(dtype=float)
    finite_ef = np.isfinite(ef_all)
    if finite_ef.any():
        ef_min = float(np.min(ef_all[finite_ef]))
        ef_max = float(np.max(ef_all[finite_ef]))
        pad = (ef_max - ef_min) * 0.05 if ef_max > ef_min else (0.05 * (abs(ef_max) if ef_max != 0 else 1.0))
        ef_ylim = (ef_min - pad, ef_max + pad)
    else:
        ef_ylim = (0.0, 1.0)

    # Prepare subplots
    group_names = sorted(pd.unique(df['Group'].astype(str)))
    G = len(group_names)
    if G == 0:
        raise ValueError("No groups could be derived from `meta` index prefixes.")
    ncols_eff = max(1, min(int(ncols), G))
    nrows = int(np.ceil(G / ncols_eff))
    # dynamic fig size
    width_per_col = 6.0
    height_per_row = 4.0
    fig, axes = plt.subplots(nrows, ncols_eff, figsize=(width_per_col*ncols_eff, height_per_row*nrows), squeeze=False, sharey=True)

    # one left and one right y-axis per row
    axes2_grid = [[None for _ in range(ncols_eff)] for _ in range(nrows)]
    axes_used_grid = [[False for _ in range(ncols_eff)] for _ in range(nrows)]

    palette = ['olivedrab', 
               'darkorange',
               'brown', 
               'dodgerblue', 
               'purple', 
               'navy']
    colors = palette[:k]

    first_handles = None
    first_labels = None

    for gi, gname in enumerate(group_names):
        r, c = divmod(gi, ncols_eff)
        ax = axes[r, c]
        sub = (df.loc[df['Group'] == gname]
                 .sort_values('__order__', kind='mergesort')
                 .reset_index(drop=True))
        if sub.empty:
            ax.set_axis_off()
            continue

        # Bars
        sub_plot = sub.set_index('Date')[a_cols].rename(columns=archetype_label_map)
        sub_plot.plot(kind='bar', stacked=True, ax=ax, 
                      color=colors, 
                      width=0.6, legend=False)
        ax.set_title(str(gname), fontsize=fontsize)
        ax.set_ylabel('Archetype Composition', fontsize=fontsize)

        # Secondary emission line
        ax2 = ax.twinx()
        x = np.arange(len(sub))
        emission_vals = sub['EF'].to_numpy(dtype=float)
        if np.isfinite(emission_vals).any():
            ax2.plot(x, emission_vals, marker='o', color='k', markersize=5, linewidth=1.5, label=emission_col, zorder=5)
        # Defer right y-axis label to a single panel per row
        ax2.set_ylabel('')
        ax2.set_xlim(ax.get_xlim())
        ax.tick_params(axis='both', labelsize=fontsize)
        ax2.tick_params(axis='y', labelsize=fontsize)
        # Enforce shared/identical y-axes
        ax.set_ylim(*primary_ylim)
        ax2.set_ylim(*ef_ylim)
        # Record that this (row, col) is used and store its secondary axis
        axes2_grid[r][c] = ax2
        axes_used_grid[r][c] = True

        # High-emission shading within this group
        high_mask = sub['EF_class'].fillna('').eq(high_label).to_numpy()
        idxs = np.flatnonzero(high_mask)
        if idxs.size:
            splits = np.where(np.diff(idxs) > 1)[0] + 1
            for g in np.split(idxs, splits):
                start, end = int(g[0]), int(g[-1])
                ax.axvspan(start - 0.5, end + 0.5, facecolor='k', alpha=0.25, zorder=0)

        # x tick labels
        ax.set_xlabel('')
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment('right')
            tick.set_fontsize(fontsize)

        # Capture handles/labels once for global legend
        if first_handles is None:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            first_handles, first_labels = (h1 + h2, l1 + l2)

    for r in range(nrows):
        used_cols = [c for c in range(ncols_eff) if axes_used_grid[r][c]]
        if not used_cols:
            continue
        first_c = min(used_cols)
        last_c  = max(used_cols)
        for c in used_cols:
            ax = axes[r, c]
            # Left y-axis labels only on the first used column in this row
            ax.set_ylabel('Archetype Composition' if c == first_c else '', fontsize=fontsize)
            ax.tick_params(labelleft=(c == first_c), labelsize=fontsize)

            # Right y-axis labels only on the last used column in this row
            ax2 = axes2_grid[r][c]
            if ax2 is not None:
                if c == last_c:
                    ax2.set_ylabel('$N_2O$ Emission (kgN/d)', fontsize=fontsize)
                    ax2.tick_params(labelright=True, labelsize=fontsize)
                    ax2.spines['right'].set_visible(True)
                else:
                    ax2.set_ylabel('')
                    ax2.tick_params(labelright=False, labelsize=fontsize)
                    ax2.spines['right'].set_visible(False)

    # Hide any unused axes
    for gi in range(G, nrows*ncols_eff):
        r, c = divmod(gi, ncols_eff)
        axes[r, c].set_axis_off()

    # Global legend outside top-right
    if first_handles:
        fig.legend(first_handles, first_labels,
                   loc='upper left', bbox_to_anchor=(1.0, 1.0),
                   frameon=False, fontsize=fontsize)

    fig.tight_layout()
    plt.show()

    return df


def plot_archetype_profiles_heatmap(X,
                                    n_archetypes: int = 3,
                                    tolerance: float = 1e-4,
                                    max_iter: int = 1000,
                                    n_init: int = 5,
                                    init: str = 'furthest_sum',
                                    top_n: int = 20,
                                    random_state: int = 42,
                                    sqrt_: bool = False,
                                    log_scale: bool = False,
                                    cmap: str = 'viridis',
                                    annotate: bool = False,
                                    fontsize: int = 9,
                                    figsize: tuple[float, float] | None = None,
                                    archetype_prefix: str = 'A'):
    """
    Heatmap of genus composition for each archetype.

    Rows = genera, columns = archetypes (A1..Ak). Values are the archetype profiles
    (aa.archetypes_) restricted to the top-N genera by mean abundance across archetypes.

    Parameters
    ----------
    X : pd.DataFrame
        Rows = samples, columns = genera (relative abundances).
    n_archetypes : int, default 3
        Number of archetypes to fit.
    tolerance : float, default 1e-4
        Convergence tolerance for archetype fitting.
    max_iter : int, default 1000
        Maximum number of iterations for archetype fitting.
    n_init : int, default 5
        Number of random initializations for archetype fitting.
    init : str, default 'furthest_sum'
        Initialization method for archetype fitting.
    top_n : int, default 20
        Number of top genera (by mean abundance across archetypes) to show.
    random_state : int, default 42
        Seed for reproducibility.
    sqrt_ : bool, default False
        Whether to sqrt-transform values for better visibility.
    log_scale : bool, default False
        If True, plot log10(values). Values at 0 are clipped to a small epsilon.
    cmap : str, default 'viridis'
        Colormap for the heatmap.
    annotate : bool, default False
        If True, overlay numeric values in each heatmap cell.
    fontsize : int, default 9
        Font size for tick labels.
    figsize : tuple[float, float] | None, default None
        Figure size; if None, a reasonable size is inferred from (top_n, n_archetypes).
    archetype_prefix : str, default='A'
        Prefix used for archetype labels on the plot (e.g. 'W' or 'U').

    Returns
    -------
    fig : matplotlib.figure.Figure
        The heatmap figure.
    H : pd.DataFrame
        Heatmap table (rows = genera, cols = archetypes).
    aa : fitted AA object
        The fitted archetype model.
    """

    # Column names (genera)
    genus_names = [c for c in X.columns]
    p = len(genus_names)
    top_n = max(1, min(int(top_n), p))

    # Fit archetypes on the raw values
    aa = AA(n_archetypes=n_archetypes, random_state=random_state,
            tol=tolerance, max_iter=max_iter, n_init=n_init, init=init).fit(X.values)

    A = aa.archetypes_  # shape (k, p)
    if A.shape[1] != p:
        raise RuntimeError(f"Expected aa.archetypes_ to have {p} features, got {A.shape[1]}")

    # Choose top-N genera by mean abundance across archetypes
    mean_abund = A.mean(axis=0)
    idx = np.argsort(mean_abund)[::-1][:top_n]

    # Clean labels (strip g__ prefix)
    labels = np.array(genus_names)[idx]
    labels = np.array([s[3:] if isinstance(s, str) and s.startswith('g__') else s for s in labels])

    # Heatmap matrix: rows = genera, cols = archetypes
    Hvals = A[:, idx].T  # (top_n, k)

    # Optional transforms
    eps = 1e-12
    if sqrt_:
        Hvals = np.sqrt(Hvals)
    if log_scale:
        Hvals = np.log10(np.clip(Hvals, eps, None))

    # DataFrame for downstream use
    colnames = _archetype_labels_long_default(n_archetypes, archetype_prefix)
    H = pd.DataFrame(Hvals, index=labels, columns=colnames)

    # Figure size heuristic
    if figsize is None:
        width = max(4.0, 1.0 + 1.0 * n_archetypes)
        height = max(3.0, 1.6 + 0.23 * top_n)
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(H.values, aspect='auto', interpolation='nearest', cmap=cmap)

    # Ticks/labels
    ax.set_xticks(np.arange(n_archetypes))
    ax.set_xticklabels(colnames, fontsize=fontsize)

    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels(H.index.tolist(), fontsize=fontsize)

    # ax.set_xlabel('Archetypes', fontsize=fontsize + 1)
    # ax.set_ylabel('Genus', fontsize=fontsize + 1)

    # Optional cell annotations
    if annotate:
        # Avoid clutter for large heatmaps
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                ax.text(j, i, f"{H.iat[i, j]:.3f}", ha='center', va='center', fontsize=max(6, fontsize - 2))

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    if log_scale:
        cbar.set_label('log(Abundance)', fontsize=fontsize)
    elif sqrt_:
        cbar.set_label('sqrt(Abundance)', fontsize=fontsize)
    else:
        cbar.set_label('Abundance', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=max(6, fontsize - 1))

    plt.tight_layout()
    plt.show()

    return fig, H, aa


def plot_coefficients_vs_continuous_panels(aa, meta,
                                           sample_index=None,
                                           col: str = 'EF',
                                           ncols: int = 3,
                                           figsize_per_panel: tuple[float, float] = (3.0, 3.2),
                                           alpha: float = 0.75,
                                           s: int = 45,
                                           add_trend: bool = False,
                                           trend_deg: int = 1,
                                           fontsize: int = 12,
                                           legend_height: float = 0.7,
                                           class_col: str = 'class',
                                           multiple_reactors: bool = True,
                                           archetype_prefix: str = 'A'):
    """Plot continuous variable vs archetype coefficients in N panels.

    This function uses the **same alignment conventions** as `plot_simplex_continuous`:
    - If `sample_index` is provided, `meta[col]` is aligned by index via `reindex(sample_index)`.
    - Else if `len(meta) == n_samples`, assumes positional alignment.
    - Otherwise all values are treated as missing.

    Points are styled as follows:
    - color encodes emission class from `meta[class_col]` (e.g. high/low emission)
    - marker encodes reactor/group from the sample index prefix when `multiple_reactors=True`

    Parameters
    ----------
    aa : fitted AA object
        Must have attribute ``coefficients_`` with shape (n_samples, n_archetypes).
    meta : pandas.DataFrame
        Must contain a column with the continuous variable (`col`).
    sample_index : array-like, optional
        Index of the samples used to fit AA. If provided, used to align `meta`.
    col : str, default='EF'
        Continuous variable column name in `meta`.
    ncols : int, default=3
        Number of subplot columns.
    figsize_per_panel : tuple(float, float), default=(3.0, 3.2)
        Figure size per panel (width, height). Total figure size scales with number of panels.
    alpha : float, default=0.75
        Scatter alpha.
    s : int, default=45
        Scatter marker size.
    add_trend : bool, default=False
        If True, overlays a simple polynomial trend line (degree `trend_deg`) on finite points.
    trend_deg : int, default=1
        Degree for the polynomial trend line if `add_trend=True`.
    fontsize : int, default=12
        Font size for all plot text: axis labels, titles, tick labels, and legend text.
    class_col : str, default='class'
        Column in `meta` containing emission classes used for point colors.
    multiple_reactors : bool, default=True
        If True, reactor/group markers are derived from the sample index prefix
        (substring before the first underscore). If False, all samples are treated
        as belonging to a single reactor/group and use the same marker.
    archetype_prefix : str, default='A'
        Prefix used for archetype labels on the plot (e.g. 'W' or 'U').

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : np.ndarray
        Array of axes (length = n_archetypes).
    """

    coeffs = np.asarray(getattr(aa, 'coefficients_', None))
    if coeffs is None:
        raise ValueError("`aa` must have attribute `coefficients_`.")

    n, k = coeffs.shape
    archetype_labels = _archetype_labels(k, archetype_prefix)

    # ---- Continuous values aligned to samples (same logic as plot_simplex_continuous) ----
    if isinstance(meta, pd.DataFrame) and (col in meta.columns):
        if sample_index is not None:
            cont = pd.to_numeric(meta[col], errors='coerce').reindex(sample_index)
        elif len(meta) == n:
            cont = pd.to_numeric(meta[col], errors='coerce').reset_index(drop=True)
        else:
            cont = pd.Series([np.nan] * n)
    else:
        cont = pd.Series([np.nan] * n)

    y = cont.to_numpy(dtype=float)
    finite = np.isfinite(y)

    # ---- Emission classes (for colors) ----
    classes = None
    class_names = np.array([])
    classes_arr = None
    if isinstance(meta, pd.DataFrame) and (class_col in meta.columns):
        if sample_index is not None:
            classes = meta[class_col].reindex(sample_index)
        elif len(meta) == n:
            classes = meta[class_col].reset_index(drop=True)
        else:
            classes = pd.Series([np.nan] * n)

        labeled_mask = classes.notna() & (classes.astype(str).str.strip() != '')
        class_names = np.unique(classes[labeled_mask].astype(str).values)
        classes_arr = classes.astype(str).values

    # ---- Reactor groups for markers ----
    if multiple_reactors:
        if sample_index is not None:
            idx_for_groups = pd.Index(sample_index)
        elif isinstance(meta, pd.DataFrame) and len(meta) == n:
            idx_for_groups = meta.index
        else:
            idx_for_groups = pd.RangeIndex(n)

        groups_series = pd.Index(idx_for_groups).astype(str).to_series(index=np.arange(n))
        groups_series = groups_series.str.split('_', n=1, expand=False).str[0]
        groups_series = groups_series.fillna('Unknown').astype(str)
    else:
        groups_series = pd.Series(['Samples'] * n, index=np.arange(n), dtype='object')

    groups_arr = groups_series.values
    group_names = np.unique(groups_arr)

    marker_cycle = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', 'h', 'H', 'd', 'p']
    group_to_marker = {
        name: marker_cycle[i % len(marker_cycle)] for i, name in enumerate(group_names)
    }

    cmap = ListedColormap([
        'tomato',
        'steelblue',
        'olivedrab',
        'darkorange',
        'brown',
        'dodgerblue',
        'purple',
        'navy'
    ])

    # ---- Layout ----
    ncols_eff = max(1, min(int(ncols), k))
    nrows = int(np.ceil(k / ncols_eff))

    fig_w = figsize_per_panel[0] * ncols_eff
    fig_h = figsize_per_panel[1] * nrows
    fig, axes_grid = plt.subplots(
        nrows, ncols_eff,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex='col',
        sharey='row',
    )

    # Flatten for easy indexing; we will return only the first k axes
    axes = axes_grid.ravel()

    # Nice label for y
    ylab = '$N_2O$ Emission Factor (-)' if str(col).strip().lower() == 'ef' else str(col)

    # ---- Plot each archetype panel ----
    for j in range(k):
        ax = axes[j]
        x = coeffs[:, j]

        # Finite y values: color by class and marker by reactor
        if finite.any():
            if classes_arr is not None and class_names.size:
                for i, cname in enumerate(class_names):
                    for g in group_names:
                        mask = finite & (classes_arr == cname) & (groups_arr == g)
                        if np.any(mask):
                            ax.scatter(
                                x[mask], y[mask],
                                s=s, alpha=alpha,
                                color=cmap(i),
                                marker=group_to_marker[g],
                                edgecolor='none'
                            )
            else:
                for g in group_names:
                    mask = finite & (groups_arr == g)
                    if np.any(mask):
                        ax.scatter(
                            x[mask], y[mask],
                            s=s, alpha=alpha,
                            color='grey',
                            marker=group_to_marker[g],
                            edgecolor='none'
                        )

        # Missing y values: grey but still keep reactor marker
        if (~finite).any():
            base = float(np.nanmin(y[finite])) if finite.any() else 0.0
            jitter = (np.random.RandomState(0).rand(np.sum(~finite)) - 0.5) * 0.02
            miss_idx = np.flatnonzero(~finite)
            for g in group_names:
                gmask = (~finite) & (groups_arr == g)
                if np.any(gmask):
                    local_idx = np.searchsorted(miss_idx, np.flatnonzero(gmask))
                    ax.scatter(
                        x[gmask], base + jitter[local_idx],
                        s=max(15, int(s * 0.6)), alpha=0.45,
                        color='lightgrey',
                        marker=group_to_marker[g],
                        edgecolor='none'
                    )

        # Optional trend line
        if add_trend and finite.any() and np.sum(finite) >= max(3, trend_deg + 1):
            try:
                order = np.argsort(x[finite])
                xf = x[finite][order]
                yf = y[finite][order]
                p = np.polyfit(xf, yf, deg=int(trend_deg))
                xx = np.linspace(float(np.min(xf)), float(np.max(xf)), 200)
                yy = np.polyval(p, xx)
                ax.plot(xx, yy, linewidth=1.5, color='grey', alpha=0.7, linestyle='--')
            except Exception:
                pass

        ax.set_title(archetype_labels[j], fontsize=fontsize, fontweight='bold')
        ax.set_xlabel(f"Coefficient ({archetype_labels[j]})", fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)
        ax.grid(True, ls=':', alpha=0.6)
        ax.set_box_aspect(1)

        # Only left-most column shows y-label to reduce clutter
        if (j % ncols_eff) == 0:
            ax.set_ylabel(ylab, fontsize=fontsize)
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

        # Only bottom row shows x tick labels to reduce clutter with shared x-axes
        if (j // ncols_eff) < (nrows - 1):
            ax.tick_params(labelbottom=False)

    # Hide any extra axes (if k is not a multiple of ncols)
    for j in range(k, len(axes)):
        axes[j].axis('off')

    # ---- Combined legend: classes by color, reactors by marker ----
    handles = []
    if classes_arr is not None and class_names.size:
        class_handles = [
            Patch(
                facecolor=cmap(i),
                edgecolor=cmap(i),
                alpha=alpha,
                label=str(cname),
            )
            for i, cname in enumerate(class_names)
        ]
        handles.extend(class_handles)

    if multiple_reactors:
        group_handles = [
            Line2D(
                [0], [0],
                marker=group_to_marker[g],
                color='grey',
                markerfacecolor='grey',
                markeredgecolor='k',
                linestyle='None',
                markersize=7,
                label=str(g),
            )
            for g in group_names
        ]
    else:
        group_handles = [
            Line2D(
                [0], [0],
                marker=group_to_marker[group_names[0]],
                color='grey',
                markerfacecolor='grey',
                markeredgecolor='k',
                linestyle='None',
                markersize=7,
                label='Samples',
            )
        ]
    handles.extend(group_handles)

    if (~finite).any():
        handles.append(
            Line2D(
                [0], [0],
                marker='o',
                color='lightgrey',
                markerfacecolor='lightgrey',
                markeredgecolor='lightgrey',
                linestyle='None',
                markersize=7,
                label=f'{col} missing',
            )
        )

    if handles:
        fig.legend(
            handles=handles,
            loc='center left',
            bbox_to_anchor=(1.01, legend_height),
            frameon=False,
            fontsize=fontsize,
        )
        fig.subplots_adjust(right=0.82)

    fig.tight_layout()
    plt.show()