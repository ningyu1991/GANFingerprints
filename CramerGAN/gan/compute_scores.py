from __future__ import division, print_function

import os.path, sys, tarfile
import numpy as np
from scipy import linalg
from six.moves import range, urllib
from sklearn.metrics.pairwise import polynomial_kernel
import tensorflow as tf
from tqdm import tqdm


# from tqdm docs: https://pypi.python.org/pypi/tqdm#hooks-and-callbacks
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # also sets self.n = b * bsize


class Inception(object):
    def __init__(self):
        MODEL_DIR = '/tmp/imagenet'
        DATA_URL = ('http://download.tensorflow.org/models/image/imagenet/'
                    'inception-2015-12-05.tgz')
        self.softmax_dim = 1008
        self.coder_dim = 2048

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(filepath):
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc=filename) as t:
                filepath, _ = urllib.request.urlretrieve(
                    DATA_URL, filepath, reporthook=t.update_to)

        tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
        with tf.gfile.FastGFile(os.path.join(
                MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        # Works with an arbitrary minibatch size.
        self.sess = sess = tf.Session()
        #with sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = [s.value for s in o.get_shape()]
                if len(shape) and shape[0] == 1:
                    shape[0] = None
                o._shape = tf.TensorShape(shape)
        w = sess.graph.get_operation_by_name(
            "softmax/logits/MatMul").inputs[1]
        self.coder = tf.squeeze(tf.squeeze(pool3, 2), 1)
        logits = tf.matmul(self.coder, w)
        self.softmax = tf.nn.softmax(logits)

        assert self.coder.get_shape()[1].value == self.coder_dim
        assert self.softmax.get_shape()[1].value == self.softmax_dim

        self.input = 'ExpandDims:0'


class LeNet(object):
    def __init__(self):
        MODEL_DIR = 'lenet/saved_model'
        self.softmax_dim = 10
        self.coder_dim = 512

        self.sess = sess = tf.Session()

        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.TRAINING], MODEL_DIR)
        g = sess.graph

        self.softmax = g.get_tensor_by_name('Softmax_1:0')
        self.coder = g.get_tensor_by_name('Relu_5:0')

        assert self.coder.get_shape()[1].value == self.coder_dim
        assert self.softmax.get_shape()[1].value == self.softmax_dim
        self.input = 'Placeholder_2:0'


def featurize(images, model, batch_size=100, transformer=np.asarray,
              get_preds=True, get_codes=False, output=sys.stdout, 
              out_preds=None, out_codes=None):
    '''
    images: a list of numpy arrays with values in [0, 255]
    '''
    sub = transformer(images[:10])
    assert(sub.ndim == 4)
    if isinstance(model, Inception):
        assert sub.shape[3] == 3
        if (sub.max() > 255) or (sub.min() < 0):
            print('WARNING! Inception min/max violated: min = %f, max = %f. Clipping values.' % (sub.min(), sub.max()))
            sub = sub.clip(0., 255.)
    elif isinstance(model, LeNet):
        batch_size = 64
        assert sub.shape[3] == 1
        if (sub.max() > .5) or (sub.min() < -.5):
            print('WARNING! LeNet min/max violated: min = %f, max = %f. Clipping values.' % (sub.min(), sub.max()))
            sub = sub.clip(-.5, .5)

    n = len(images)

    to_get = ()
    ret = ()
    if get_preds:
        to_get += (model.softmax,)
        if out_preds is not None:
            assert out_preds.shape == (n, model.softmax_dim)
            assert out_preds.dtype == np.float32
            preds = out_preds
        else:
            preds = np.empty((n, model.softmax_dim), dtype=np.float32)
            preds.fill(np.nan)
        ret += (preds,)
    if get_codes:
        to_get += (model.coder,)
        if out_codes is not None:
            assert out_codes.shape == (n, model.coder_dim)
            assert out_codes.dtype == np.float32
            codes = out_codes
        else:
            codes = np.empty((n, model.coder_dim), dtype=np.float32)
            codes.fill(np.nan)
        ret += (codes,)

    # with model.sess:
    with TqdmUpTo(unit='img', unit_scale=True, total=n, file=output) as t:
        for start in range(0, n, batch_size):
            t.update_to(start)
            end = min(start + batch_size, n)
            inp = transformer(images[start:end])

            if end - start != batch_size:
                pad = batch_size - (end - start)
                extra = np.zeros((pad,) + inp.shape[1:], dtype=inp.dtype)
                inp = np.r_[inp, extra]
                w = slice(0, end - start)
            else:
                w = slice(None)

            out = model.sess.run(to_get, {model.input: inp})
            if get_preds:
                preds[start:end] = out[0][w]
            if get_codes:
                codes[start:end] = out[-1][w]
        t.update_to(n)
    return ret


def get_splits(n, splits=10, split_method='openai'):
    if split_method == 'openai':
        return [slice(i * n // splits, (i + 1) * n // splits)
                for i in range(splits)]
    elif split_method == 'bootstrap':
        return [np.random.choice(n, n) for _ in range(splits)]
    else:
        raise ValueError("bad split_method {}".format(split_method))


def inception_score(preds, **split_args):
    split_inds = get_splits(preds.shape[0], **split_args)
    scores = np.zeros(len(split_inds))
    for i, inds in enumerate(split_inds):
        part = preds[inds]
        kl = part * (np.log(part) - np.log(np.mean(part, 0, keepdims=True)))
        kl = np.mean(np.sum(kl, 1))
        scores[i] = np.exp(kl)
    return scores


def fid_score(codes_g, codes_r, eps=1e-6, output=sys.stdout, **split_args):
    splits_g = get_splits(codes_g.shape[0], **split_args)
    splits_r = get_splits(codes_r.shape[0], **split_args)
    assert len(splits_g) == len(splits_r)
    d = codes_g.shape[1]
    assert codes_r.shape[1] == d

    scores = np.zeros(len(splits_g))
    with tqdm(splits_g, desc='FID', file=output) as bar:
        for i, (w_g, w_r) in enumerate(zip(bar, splits_r)):
            part_g = codes_g[w_g]
            part_r = codes_r[w_r]

            mn_g = part_g.mean(axis=0)
            mn_r = part_r.mean(axis=0)

            cov_g = np.cov(part_g, rowvar=False)
            cov_r = np.cov(part_r, rowvar=False)

            covmean, _ = linalg.sqrtm(cov_g.dot(cov_r), disp=False)
            if not np.isfinite(covmean).all():
                cov_g[range(d), range(d)] += eps
                cov_r[range(d), range(d)] += eps
                covmean = linalg.sqrtm(cov_g.dot(cov_r))

            scores[i] = np.sum((mn_g - mn_r) ** 2) + (
                np.trace(cov_g) + np.trace(cov_r) - 2 * np.trace(covmean))
            bar.set_postfix({'mean': scores[:i+1].mean()})
    return scores


def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                            ret_var=True, output=sys.stdout, **kernel_args):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
        for i in bar:
            g = codes_g[choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[choice(len(codes_r), subset_size, replace=False)]
            o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
            bar.set_postfix({'mean': mmds[:i+1].mean()})
    return (mmds, vars) if ret_var else mmds


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('samples')
    parser.add_argument('reference_feats', nargs='?')
    parser.add_argument('--output', '-o')

    parser.add_argument('--reference-subset', default=slice(None),
                        type=lambda x: slice(*(int(s) if s else None
                                               for s in x.split(':'))))

    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--model', choices=['inception', 'lenet'],
                        default='inception')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--save-codes')
    g.add_argument('--load-codes')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--save-preds')
    g.add_argument('--load-preds')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--do-inception', action='store_true', default=True)
    g.add_argument('--no-inception', action='store_false', dest='do_inception')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--do-fid', action='store_true', default=False)
    g.add_argument('--no-fid', action='store_false', dest='do_fid')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--do-mmd', action='store_true', default=False)
    g.add_argument('--no-mmd', action='store_false', dest='do_mmd')
    parser.add_argument('--mmd-degree', type=int, default=3)
    parser.add_argument('--mmd-gamma', type=float, default=None)
    parser.add_argument('--mmd-coef0', type=float, default=1)

    parser.add_argument('--mmd-subsets', type=int, default=100)
    parser.add_argument('--mmd-subset-size', type=int, default=1000)
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--mmd-var', action='store_true', default=False)
    g.add_argument('--no-mmd-var', action='store_false', dest='mmd_var')

    parser.add_argument('--splits', type=int, default=10)
    parser.add_argument('--split-method', choices=['openai', 'bootstrap'],
                        default='bootstrap')

    args = parser.parse_args()

    if args.do_fid and args.reference_feats is None:
        parser.error("Need REFERENCE_FEATS if you're doing FID")

    def check_path(pth):
        if os.path.exists(pth):
            parser.error("Path {} already exists".format(pth))
        d = os.path.dirname(pth)
        if d and not os.path.exists(d):
            os.makedirs(d)

    if args.output:
        check_path(args.output)

    samples = np.load(args.samples, mmap_mode='r')

    if args.model == 'inception':
        model = Inception()
        if samples.dtype == np.uint8:
            transformer = np.asarray
        elif samples.dtype == np.float32:
            m = samples[:10].max()
            assert .5 <= m <= 1
            transformer = lambda x: x * 255
        else:
            raise TypeError("don't know how to handle {}".format(samples.dtype))
    elif args.model == 'lenet':
        model = LeNet()
        if samples.dtype == np.uint8:
            def transformer(x):
                return (np.asarray(x, dtype=np.float32) - (255 / 2.)) / 255
        elif samples.dtype == np.float32:
            assert .8 <= samples[:10].max() <= 1
            assert 0 <= samples[:10].min() <= .3
            transformer = lambda x: x - .5
        else:
            raise TypeError("don't know how to handle {}".format(samples.dtype))
    else:
        raise ValueError("bad model {}".format(args.model))

    if args.reference_feats:
        ref_feats = np.load(args.reference_feats, mmap_mode='r')[
                args.reference_subset]

    out_kw = {}
    if args.save_codes:
        check_path(args.save_codes)
        out_kw['out_codes'] = np.lib.format.open_memmap(
            args.save_codes, mode='w+', dtype=np.float32,
            shape=(samples.shape[0], model.coder_dim))
    if args.save_preds:
        check_path(args.save_preds)
        out_kw['out_preds'] = np.lib.format.open_memmap(
            args.save_preds, mode='w+', dtype=np.float32,
            shape=(samples.shape[0], model.softmax_dim))

    need_preds = args.do_inception or args.save_preds
    need_codes = args.do_fid or args.do_mmd or args.save_codes

    print('Transformer test: transformer([-1, 0, 10.]) = ' + repr(transformer(np.array([-1, 0, 10.]))))

    if args.load_codes or args.load_preds:
        if args.load_codes:
            codes = np.load(args.load_codes, mmap_mode='r')
            assert codes.ndim == 2
            assert codes.shape[0] == samples.shape[0]
            assert codes.shape[1] == model.coder_dim

        if args.load_preds:
            preds = np.load(args.load_preds, mmap_mode='r')
            assert preds.ndim == 2
            assert preds.shape[0] == samples.shape[0]
            assert preds.shape[1] == model.softmax_dim
        elif need_preds:
            raise NotImplementedError()
    else:
        out = featurize(
            samples, model, batch_size=args.batch_size, transformer=transformer,
            get_preds=need_preds, get_codes=need_codes, **out_kw)
        if need_preds:
            preds = out[0]
        if need_codes:
            codes = out[-1]

    split_args = {'splits': args.splits, 'split_method': args.split_method}

    output = {'args': args}

    if args.do_inception:
        output['inception'] = scores = inception_score(preds, **split_args)
        print("Inception mean:", np.mean(scores))
        print("Inception std:", np.std(scores))
        print("Inception scores:", scores, sep='\n')

    if args.do_fid:
        output['fid'] = scores = fid_score(codes, ref_feats, **split_args)
        print("FID mean:", np.mean(scores))
        print("FID std:", np.std(scores))
        print("FID scores:", scores, sep='\n')
        print()

    if args.do_mmd:
        ret = polynomial_mmd_averages(
            codes, ref_feats, degree=args.mmd_degree, gamma=args.mmd_gamma,
            coef0=args.mmd_coef0, ret_var=args.mmd_var,
            n_subsets=args.mmd_subsets, subset_size=args.mmd_subset_size)
        if args.mmd_var:
            output['mmd2'], output['mmd2_var'] = mmd2s, vars = ret
        else:
            output['mmd2'] = mmd2s = ret
        print("mean MMD^2 estimate:", mmd2s.mean())
        print("std MMD^2 estimate:", mmd2s.std())
        print("MMD^2 estimates:", mmd2s, sep='\n')
        print()
        if args.mmd_var:
            print("mean Var[MMD^2] estimate:", vars.mean())
            print("std Var[MMD^2] estimate:", vars.std())
            print("Var[MMD^2] estimates:", vars, sep='\n')
            print()

    if args.output:
        np.savez(args.output, **output)


if __name__ == '__main__':
    main()
