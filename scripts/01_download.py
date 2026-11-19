"""Stage 1: download MVTec AD from MVTec's servers (all categories, or the ones given)."""

import argparse

import _common  # noqa: F401
from defects import config, data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("categories", nargs="*", default=config.CATEGORIES)
    for c in ap.parse_args().categories:
        print(data.download(c))
