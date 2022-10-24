import os

DATADIR = "data"
MAX_NEG_GENERATION = 10 * 1000 * 1000
MAX_ITER = 10 * 1000 * 1000

HG_DIR = "HG"
RESULTS_DIR = "Results"

default_save_dir = os.path.join(HG_DIR, "{}", "hedge-{}", "negtype-{}-{}", "seed-{}")
default_save_path = os.path.join(default_save_dir, "hypergraph.data")

default_results_dir = os.path.join(RESULTS_DIR, "hedge-{}", "negtype-{}-{}", "seed-{}")

GNS = ["email-Eu",
       "email-Enron"]
"""
GNS = [name for name in os.listdir(DATADIR) if os.path.isdir(os.path.join(DATADIR, name))]
GNS.remove('coauth-MAG-Geology')
GNS.remove('coauth-MAG-History')
GNS.remove('coauth_DBLP')
#GNS.remove('coauth-MAG-Geology')
"""

HEDGE_SIZES = [4, 5]
IMBS = [1, 2, 5, 10]
NEG_TYPES = ["clique", "star", "tailed"]
SEED = 9
FEATURES = [
    "gm",
    "hm",
    "am",
    "cn",
    "jc",
    "aa"]
