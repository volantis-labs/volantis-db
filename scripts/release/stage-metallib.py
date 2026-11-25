#!/usr/bin/env python3
"""Stage the MLX runtime resource from this Cargo release build, never a global cache."""
import argparse
from pathlib import Path
import shutil


def stage(release, destination):
    candidates = sorted(release.glob('build/mlx-sys-*/out/build/lib/mlx.metallib'),
                        key=lambda path: path.stat().st_mtime_ns)
    if not candidates:
        raise ValueError(f'expected an MLX metallib in {release}, found none')
    # A restored CI cache can hold build-script output dirs from earlier
    # RUSTFLAGS; the newest one is the build that just ran.
    chosen = candidates[-1]
    if chosen.stat().st_size == 0:
        raise ValueError(f'MLX metallib is empty: {chosen}')
    for stale in candidates[:-1]:
        print(f'MLX runtime: ignoring older build output {stale}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(chosen, destination)
    print(f'MLX runtime: {chosen} -> {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    stage(args.release, args.destination)
