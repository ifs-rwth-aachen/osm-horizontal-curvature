import pandas as pd
from scipy import optimize, stats
import numpy as np
import os
import string
import matplotlib.pyplot as plt
import pwlf
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import tikzplotlib as tl
from sklearn.ensemble import AdaBoostRegressor
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy import interpolate
import splipy as sp
import splipy.curve_factory   as curve_factory
import scipy.interpolate as si

path = r'C:/Users/Tobias/Documents/devel/tex/ieee/ieee_paper/editorial/resultdata'
ref = pd.read_csv(os.path.join(path, 'results_track_a_ref.csv'))
osm = pd.read_csv(os.path.join(path, 'results_track_a_osm.csv'))


def decision_tree_regression(df, n_seg, model='ada'):
    xs = df['s'].to_numpy()
    ys = df['curv_hor'].to_numpy()
    dys = np.gradient(ys, xs)

    fig, (ax0, ax1) = plt.subplots(1, 2)

    if model == 'dtc':
        rgr = DecisionTreeRegressor(max_leaf_nodes=n_seg)
    elif model == 'ada':
        rgr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                n_estimators=100)
    else:
        raise ValueError('bad input for model')

    rgr.fit(xs.reshape(-1, 1), dys.reshape(-1, 1))
    dys_dt = rgr.predict(xs.reshape(-1, 1)).flatten()

    ys_sl = np.ones(len(xs)) * np.nan
    for y in np.unique(dys_dt):
        msk = dys_dt == y
        lin_reg = LinearRegression()
        lin_reg.fit(xs[msk].reshape(-1, 1), ys[msk].reshape(-1, 1))
        ys_sl[msk] = lin_reg.predict(xs[msk].reshape(-1, 1)).flatten()
        ax0.plot([xs[msk][0], xs[msk][-1]],
                 [ys_sl[msk][0], ys_sl[msk][-1]],
                 color='r', zorder=1)

    ax0.set_title('values')
    ax0.scatter(xs, ys, label='data')
    ax0.scatter(xs, ys_sl, s=3 ** 2, label='seg lin reg', color='g', zorder=5)
    ax0.legend()

    ax1.set_title('slope')
    ax1.scatter(xs, dys, label='data')
    ax1.scatter(xs, dys_dt, label='DecisionTree', s=2 ** 2)
    ax1.legend()


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])


def evaluate_piecewise(df, ref, count=8):
    x = df['s'].to_numpy()
    y = df['curv_hor'].to_numpy()
    # p, e = optimize.curve_fit(piecewise_linear, x, y)
    # xd = np.linspace(0, 15, 100)
    # plt.plot(x, y, "o")
    # plt.plot(xd, piecewise_linear(xd, *p))
    # px, py = segments_fit(x, y, count)
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit(count)
    print(breaks)

    x_hat = np.linspace(x.min(), x.max(), 100)
    y_hat = my_pwlf.predict(x_hat)
    plt.plot(x, y, "o")
    plt.plot(x_hat, y_hat, '.-', label='pwl')
    plt.plot(ref['s'], ref['curv_hor'], label='ref')
    plt.legend()


def segments_fit(X, Y, count):
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2) ** 2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)


##########################################


def distance_lower_than_threshold(ds_i, ds_thresh):
    if ds_i <= ds_thresh:
        return True
    else:
        return False


def get_slope(x, y):
    def objective(x, a, b):
        return a * x + b

    popt, _ = optimize.curve_fit(objective, x, y)
    slope, intercept = popt
    # print('y = %.5f * x + %.5f' % (slope, intercept))
    return slope, intercept


def iter_osm_data_piecewise_identify(df, ds_thresh=50):
    s = []
    c = []
    resid_s = []
    resid_c = []
    slope_temp = np.nan
    nseg = 0
    for i, row in df.iterrows():
        if i > 0:
            s_i = row['s']
            s_prev = df.loc[i - 1, 's']
            ds_i = s_i - s_prev
            c_i = row['curv_hor']
            if distance_lower_than_threshold(ds_i, ds_thresh):
                resid_s.append(s_i)
                resid_c.append(c_i)
                if len(resid_s) > 1:
                    slope_temp, _ = get_slope(resid_s, resid_c)
                    print(slope_temp)
            else:
                resid_s = []
                resid_c = []
                slope_temp = np.nan
                print(f'ds to big: {ds_i}')
                nseg += 1
            print(i, s_i, ds_i, c_i, 'points in residuum: {}'.format(len(resid_s)), f'slope: {slope_temp}')


def rolling_median(df, sthresh=50):
    x = df['s'].to_numpy()
    xmin = x.min()
    xmax = x.max()

    if xmin > 0:
        x -= xmin
        xoffset = True
    else:
        xoffset = False

    y = df['curv_hor'].to_numpy()
    df_res = pd.DataFrame({'s': np.arange(x.min(), x.max(), 1)})
    df_res['curv_hor'] = np.nan

    niter = round(xmax / sthresh)

    for i in range(niter):
        si_low = i * sthresh
        si_high = (i + 1) * sthresh
        slice = df.loc[(df['s'] >= si_low) & (df['s'] < si_high), 'curv_hor']
        slice_count = slice.count()
        df_res.loc[(df_res['s'] >= si_low) & (df_res['s'] < si_high), 'curv_hor'] = slice.median()

        # if slice_count >= 3:
        #     df_res.loc[(df_res['s'] >= si_low) & (df_res['s'] < si_high), 'curv_hor'] = slice.median()
        #     print('count >= 3, median:', slice.median())
        # else:
        #     subniter = round(slice_count / 3)
        #     for j in range(subniter):
        #         sj_low = si_low + j*5
        #         sj_high = sj_low + 5
        #         subslice = df.loc[(df['s'] >= sj_low) & (df['s'] < sj_high), 'curv_hor']
        #         if len(subslice) < 1:
        #             pass
        #         else:
        #             df_res.loc[(df_res['s'] >= sj_low) & (df_res['s'] < sj_high), 'curv_hor'] = subslice.mean()
        #             print(f'count < 3, slow: {sj_low}, shigh: {sj_high}, mean:', subslice.mean())
        df_res.loc[(df_res['s'] >= si_low) & (df_res['s'] < si_high), 'n_points'] = slice_count

    if xoffset:
        df_res['s'] = df_res['s'] + xmin + (sthresh / 2)
    df_res.loc[df_res['curv_hor'].abs() <= 1 / 2000, 'curv_hor'] = 0  # set curv_hor = 0 for radii >= 2000 m
    df_res = df_res[::sthresh]
    df_res['curv_hor'] = df_res['curv_hor'].interpolate(method='slinear')
    return df_res


def boxplots(df, absval=True):
    # sns.boxplot(y='track', x='curv_hor', hue='source', data=df, orient='h')
    # sns.violinplot(y='track', x='curv_hor', hue='source', data=df, orient='h', split=True,
    # scale_hue=False,
    # inner="quartile")
    if absval:
        df['curv_hor'] = df['curv_hor'].abs()
    sns.boxenplot(y='track', x='curv_hor', hue='source', data=df, orient='h', showfliers=False)
    sns.stripplot(y='track', x='curv_hor', hue='source', data=df, orient='h', size=2, alpha=.3)


def elementWiseStats(r, o):
    r['curv_hor_osm'] = 0
    r.loc[0, 'curv_hor_osm'] = o.loc[0, 'curv_hor']
    for i in range(1, len(r)):
        s1 = r.loc[i - 1, 's']
        s2 = r.loc[i, 's']
        c1 = r.loc[i - 1, 'curv_hor']
        c2 = r.loc[i, 'curv_hor']

        osm_slice = o.loc[(o['s'] >= s1) & (o['s'] < s2)]
        print('osm_slice', len(osm_slice))
        if len(osm_slice) > 0:
            slope, intercept, r_value, p_value, std_err = stats.linregress(osm_slice[['s', 'curv_hor']].to_numpy())
            ds = s2 - s1
            curv = slope * ds + r.loc[i - 1, 'curv_hor_osm']
            if curv == np.nan:
                curv = r.loc[i - 1, 'curv_hor_osm']

            if len(osm_slice) > 1:
                r.loc[i, 'curv_hor_osm'] = curv
            else:
                r.loc[i, 'curv_hor_osm'] = r.loc[i - 1, 'curv_hor_osm']

            # if abs(c2 - c1) <= 1/2000.0:  # sehr kleine Änderung bzw. konstante Krümmung
            #     curv = osm_slice['curv_hor'].median()
            #     r.loc[i, 'curv_hor_osm'] = curv
            #     print(i, curv, 'curv small ---')
            # else:
            #     if len(osm_slice) > 1:
            #         print(osm_slice[['s', 'curv_hor']].to_numpy())
            #         slope, intercept, r_value, p_value, std_err = stats.linregress(osm_slice[['s', 'curv_hor']].to_numpy())
            #         ds = s2 - s1
            #         curv = slope * ds + r.loc[i-1, 'curv_hor_osm']
            #         if curv > r.loc[i-1, 'curv_hor_osm']:
            #             curv = r.loc[i-1, 'curv_hor_osm']
            #         elif curv == np.nan:
            #             curv = r.loc[i - 1, 'curv_hor_osm']
            #         print(i, curv, 'slope')
            #         r.loc[i-1, 'curv_hor_osm'] = curv
            #     else:
            #         print('len(osm_slice) == 0')
            #         r.loc[i-1, 'curv_hor_osm'] = r.loc[i-1, 'curv_hor_osm']

    plt.plot(r['s'], r['curv_hor'], linewidth=2, label='ref')
    plt.plot(r['s'], r['curv_hor_osm'], label='osm')
    plt.plot(osm['s'], osm['curv_hor'], 'o', label='osm_data')
    plt.legend()


def iter_results():
    def _append(r, o, tr):
        r['track'] = tr
        r['source'] = 'REF'
        o['track'] = tr
        o['source'] = 'OSM'
        return r[['track', 'source', 's', 'curv_hor']].append(o[['track', 'source', 's', 'curv_hor']])

    boxdf = pd.DataFrame(columns=['track', 'source', 's', 'curv_hor'])
    for s in list(string.ascii_lowercase)[:15]:
        ref = pd.read_csv(os.path.join(path, f'results_track_{s}_ref.csv'))
        osm = pd.read_csv(os.path.join(path, f'results_track_{s}_osm.csv'))
        res = rolling_median(osm, 20)
        trackdata = _append(ref, osm, s)
        trackdata = _append(ref, res, s)
        boxdf = boxdf.append(trackdata)
        # ax = osm.plot(x='s', y='curv_hor', label='raw_osm', style='o')
        # ax = res.plot(x='s', y='curv_hor', label='processed_osm')
        # #res.fillna(method='ffill').plot(x='s', y='n_points', label='n_points', ax=ax)
        # ref.plot(x='s', y='curv_hor', label='ref', ax=ax, color='red')
        # plt.legend()
        # plt.show()
    boxplots(boxdf)


def b_spline_to_bezier_series(tck, per=False):
    """Convert a parametric b-spline into a sequence of Bezier curves of the same degree.

    Inputs:
      tck : (t,c,k) tuple of b-spline knots, coefficients, and degree returned by splprep.
      per : if tck was created as a periodic spline, per *must* be true, else per *must* be false.

    Output:
      A list of Bezier curves of degree k that is equivalent to the input spline.
      Each Bezier curve is an array of shape (k+1,d) where d is the dimension of the
      space; thus the curve includes the starting point, the k-1 internal control
      points, and the endpoint, where each point is of d dimensions.
    """
    from scipy.interpolate.fitpack import insert
    from numpy import asarray, unique, split, sum
    t, c, k = tck
    t = asarray(t)
    try:
        c[0][0]
    except:
        # I can't figure out a simple way to convert nonparametric splines to
        # parametric splines. Oh well.
        raise TypeError("Only parametric b-splines are supported.")
    new_tck = tck
    if per:
        # ignore the leading and trailing k knots that exist to enforce periodicity
        knots_to_consider = unique(t[k:-k])
    else:
        # the first and last k+1 knots are identical in the non-periodic case, so
        # no need to consider them when increasing the knot multiplicities below
        knots_to_consider = unique(t[k + 1:-k - 1])
    # For each unique knot, bring it's multiplicity up to the next multiple of k+1
    # This removes all continuity constraints between each of the original knots,
    # creating a set of independent Bezier curves.
    desired_multiplicity = k + 1
    for x in knots_to_consider:
        current_multiplicity = sum(t == x)
        remainder = current_multiplicity % desired_multiplicity
        if remainder != 0:
            # add enough knots to bring the current multiplicity up to the desired multiplicity
            number_to_insert = desired_multiplicity - remainder
            new_tck = insert(x, new_tck, number_to_insert, per)
    tt, cc, kk = new_tck
    # strip off the last k+1 knots, as they are redundant after knot insertion
    bezier_points = numpy.transpose(cc)[:-desired_multiplicity]
    if per:
        # again, ignore the leading and trailing k knots
        bezier_points = bezier_points[k:-k]
    # group the points into the desired bezier curves
    return split(bezier_points, len(bezier_points) / desired_multiplicity, axis=0)


def plot_2D_curve(curve, show_controlpoints=False):
    t = np.linspace(curve.start(), curve.end(), 150)
    x = curve(t)
    plt.plot(x[:, 0], x[:, 1])
    if (show_controlpoints):
        plt.plot(curve[:, 0], curve[:, 1], 'rs-')
    plt.axis('equal')
    plt.show()


def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count + degree + 1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree, 1, degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree, 1, count - 1)

    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0 - degree, count + degree + degree - 1, dtype='int')
    else:
        kv = np.concatenate(([0] * degree, np.arange(count - degree + 1), [count - degree] * degree))

    # Calculate query range
    u = np.linspace(periodic, (count - degree), n)

    # Calculate result
    return np.array(si.splev(u, (kv, cv.T, degree))).T


def interpolateTrack(o):
    ds = 0.1
    xs = np.linspace(0, (max(o['lon']) - min(o['lon'])), int(len(o) / ds))
    bezier = curve_factory.bezier(o[['lon', 'lat']].values)
    # plot_2D_curve(bezier, show_controlpoints=True)
    return bezier

# evaluate_piecewise(osm.head(25), ref[ref['s'] <= 301.890086], 8)
# decision_tree_regression(osm.head(25), 8)

# iter_osm_data_piecewise_identify(osm)
# iter_results()

cv = osm[['lon', 'lat']].values
plt.plot(cv[:,0], cv[:,1], '.', label='raw')
for i in range(3, 6):
    res = bspline(cv, n=10000, degree=i)
    plt.plot(res[:,0], res[:,1], label=f'spline {i}th order')
plt.legend()

