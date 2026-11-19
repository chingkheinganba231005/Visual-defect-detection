"""Paths, seeds and the assumptions behind the inspection decisions."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DEFECTS_DATA_DIR", ROOT / "data" / "mvtec"))
MODELS_DIR = Path(os.environ.get("DEFECTS_MODELS_DIR", ROOT / "models"))
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Official per-category archives of MVTec AD (MVTec Software GmbH, CC BY-NC-SA 4.0),
# with the SHA-256 of each archive recorded when it was first downloaded on 19 Nov 2026.
DOWNLOADS = {
    "bottle": ("https://www.mydrive.ch/shares/163640/9a565b9a02f45cd135e46b1c36e01ec5/download/420937370-1629958698/bottle.tar.xz",
               "726512129cb3b1f47d4f6cd7c7fc9170db7fc209132249c788fec06029cb5dc3"),
    "cable": ("https://www.mydrive.ch/shares/163641/97fd0295152453e533cfb3db1d253a35/download/420937413-1629958794/cable.tar.xz",
              "1e3d59b80a6a8da5adf32fb0cf311c7b964cde146d6920aef9ca66380999a9cd"),
    "capsule": ("https://www.mydrive.ch/shares/163642/236d4304eaf9526b350b18807789ce86/download/420937454-1629958872/capsule.tar.xz",
                "f6a41cb11f1589d552888fe9c43a1adcbcae15b70073e19093322f836d00a2b6"),
    "carpet": ("https://www.mydrive.ch/shares/163643/70e8bebbe9f2d2c55043311a4e67eb1d/download/420937484-1629959013/carpet.tar.xz",
               "d9dd5064515a20bd75cf24d223c570579141abd72388beef892780b74dbaa85d"),
    "grid": ("https://www.mydrive.ch/shares/163644/8c7fc7cba43ef5275c87ca750a6d0004/download/420937487-1629959044/grid.tar.xz",
             "d94a9a651cbfaee22c94d06cf8fa8204a590c872362de0489eb34f1645328de8"),
    "hazelnut": ("https://www.mydrive.ch/shares/163645/8346f81c9e093d2ebbb3a73f98372217/download/420937545-1629959162/hazelnut.tar.xz",
                 "a7f4e7d65a60a235def7236a46fcb854d62e6c6917c16b1ec9cb4ff089d3b0da"),
    "leather": ("https://www.mydrive.ch/shares/163646/7e0627b5718d2902bb85d24cff7a51ce/download/420937607-1629959262/leather.tar.xz",
                "704ffa48b1dc2fc44824aa50442d9febc399dd92a5d6fd8156f8aa0373d15f24"),
    "metal_nut": ("https://www.mydrive.ch/shares/163647/6d5a833cdee9035a45d5ce0921ecd24d/download/420937637-1629959294/metal_nut.tar.xz",
                  "86e3c8e163ceb19146ad3ddc408e85cc34ae02e0da430e07a73bac6bbb48532a"),
    "pill": ("https://www.mydrive.ch/shares/163648/f8c4ca0d325888f2ba128a9ed44708f5/download/420938129-1629960351/pill.tar.xz",
             "ac6ce9ec4ec2a5d67c906d82147bd7431980cf00b78f25d63201a5f17e3a2038"),
    "screw": ("https://www.mydrive.ch/shares/163649/cfd394b4e3a49013745fab9e2084988a/download/420938130-1629960389/screw.tar.xz",
              "0a27a365a21e472d19b9fb48cb8e4002b49fabe083bff7f4c16fa57d78154f10"),
    "tile": ("https://www.mydrive.ch/shares/163650/55ad2038770deff6fe6771bd784af8cc/download/420938133-1629960456/tile.tar.xz",
             "392228153b96d0db9f179d3f7c2d63e131e907154883b57309fcbf798643173c"),
    "toothbrush": ("https://www.mydrive.ch/shares/163651/f03d8421ec64b3e3bfa4a2ba2fbafb4f/download/420938134-1629960477/toothbrush.tar.xz",
                   "fe72a34b4b0d074f84e844be7ef8c4f64f7f02f20e25f19c9e3b4a231c8bea77"),
    "transistor": ("https://www.mydrive.ch/shares/163652/619a88c3d5f3f57b46b00f44f04ad6e6/download/420938166-1629960554/transistor.tar.xz",
                   "146b2166c35a1d0cf37ded091366ac01a50338b4ac704632f1239890eaca4449"),
    "wood": ("https://www.mydrive.ch/shares/163653/937e0d65bc9c4497d253cb7313868a60/download/420938383-1629960649/wood.tar.xz",
             "3757297ac2a266072eb4f25a97f805f5b37fd583492b20bd709f310736c1ea47"),
    "zipper": ("https://www.mydrive.ch/shares/163654/7f6e6aefbf00a9e8a09cc0bbac236eb0/download/420938385-1629960680/zipper.tar.xz",
               "7ab46e0e195da145db30052baea3d3a5d14c5f0e710137c968e0423f02cb942d"),
}
CATEGORIES = list(DOWNLOADS)
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}

RESIZE = 256
IMAGE_SIZE = 224
SEED = 20261119

# PatchCore
CORESET_RATIO = 0.01      # share of training patches kept in the memory bank
PROJECTION_DIM = 128      # random projection used only to pick the coreset
BLUR_SIGMA = 4.0          # smoothing of the anomaly map, in pixels at 224 x 224

# Each category's test set is split in two, stratified by defect type: thresholds
# are chosen on the calibration half and judged on the held-out half.
CALIBRATION_SHARE = 0.5
N_BOOTSTRAP = 1000

# Inspection economics (assumptions, varied in the sensitivity analysis)
DEFECT_RATE = 0.02        # share of production units that are defective
COST_INSPECTION = 1.0     # USD, a human inspector checking one flagged unit
COST_ESCAPE = 100.0       # USD, a defective unit reaching the customer (return, replacement, client trust)
CATCH_TARGET = 0.99       # triage policy: share of defects that must be sent to an inspector
