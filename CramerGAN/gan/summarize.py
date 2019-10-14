import argparse
import os

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
parser.add_argument('--tex', action='store_true')
args = parser.parse_args()

if args.tex:
    split = ' & '
    end = '\\\\\n'
else:
    split = '  '
    end = '\n'
    
print(' ' * 79 + 'Inception    (std)       FID    (std)     MMD^2     (std)')
print(' ' * (18 if args.tex else 87), end=split)
print('{:>9}'.format('Inception'), end=' ' if args.tex else '  ')
print('{:>7}'.format('' if args.tex else '(std)'), end=split)
print('{:>8}'.format('FID'), end='  ')
print('{:>7}'.format('' if args.tex else '(std)'), end=split)
print('{:>8}'.format('KID'), end='  ')
print('{:>8}'.format('' if args.tex else '(std)'), end=end)
if args.tex:
    print("\\hline")

for fn in sorted(args.files):
    with np.load(fn) as d:
        n = '/'.join(fn.split('/')[-3:-1])#os.path.basename(fn)
        if n.endswith('.npz'):
            n = n[:-4]
        if n.endswith('-results'):
            n = n[:-len('-results')]
        if args.tex:
            n = n.replace('_', ' ')

        print('{:88}'.format(n), end=split)
        print('{:8.3f}'.format(d['inception'].mean()), end='  ')
        print('({:5.3f})'.format(d['inception'].std()), end=split)
        print('{:8.3f}'.format(d['fid'].mean()), end='  ')
        print('({:5.3f})'.format(d['fid'].std()), end=split)
        print('{:8.4f}'.format(d['mmd2'].mean()), end='  ')
        print('({:6.4f})'.format(d['mmd2'].std()), end=end)

