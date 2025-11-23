import os
import argparse
import subprocess
import numpy as np


def add_argparse_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--root', type=str, default='/public_bme/data/lishr/Cross_modal/subjects')
    parser.add_argument('--subject', type=str)

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_argparse_args(parser)
    args = parser.parse_args()

    subject_dir = os.path.join(args.root, args.subject, 'surf')

    # Left hemisphere
    lh_dir = os.path.join(subject_dir, 'lh.full.patch.3d')
    lh_target = os.path.join(subject_dir, 'lh.full.flat.patch.3d')
    if not os.path.exists(lh_target):
        subprocess.run([
            'mris_flatten', '-dilate', '1', '-distances', '20', '20', lh_dir, lh_target
        ])

    # Right hemisphere
    rh_dir = os.path.join(subject_dir, 'rh.full.patch.3d')
    rh_target = os.path.join(subject_dir, 'rh.full.flat.patch.3d')
    if not os.path.exists(rh_target):
        subprocess.run([
            'mris_flatten', '-dilate', '1', '-distances', '20', '20', rh_dir, rh_target
        ])