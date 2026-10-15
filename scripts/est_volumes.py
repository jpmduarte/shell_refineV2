# Writes the head volumes a size-normalised evaluation cache was built with, as the
# {case: mL} JSON that size_norm.py --volumes reads, so another grid (--factor) can be
# normalised with exactly the same pass-1 estimates.
#
#     python scripts/est_volumes.py caches/cache_norm4_est_s42 > est_s42.json

import json
import os
import sys

meta = json.load(open(os.path.join(sys.argv[1], "norm.json")))
json.dump({cid: m["volume_ml"] for cid, m in meta["cases"].items()}, sys.stdout, indent=1)
