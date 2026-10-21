# Checks that the code still does what the reported numbers were produced with.
#
#     python -m unittest discover tests -v           # everything, ~5 min on the GPU
#     python -m unittest tests.test_method.Geometry  # one group
#
# Each group checks one thing that was verified by hand during the project:
#     Geometry     padding and resampling (CPU, no data)
#     Guard        the pass-1 scale guard: a normal case, a scale out of range, an empty volume
#     Reproduce    a stored checkpoint gives the stored scores on one case
#     Segment      segment.py on a NIfTI, in RAS and in another orientation
#     Cache        a rebuilt size-normalised case is identical to the cached one
#     Train        2 seeded epochs give the known losses
# The groups that need the caches, the runs or a GPU are skipped when those are missing.

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from common.paths import NATIVE_CACHE, NORM_CACHE, SIZE_NORM_DIR, SIZE_NORM_RUNS  # noqa: E402

from common.geometry import affine_resample, spacing_grid  # noqa: E402
from size_normalization.infer import load_net, pad, segment, segment_plain  # noqa: E402

CACHE = NATIVE_CACHE
NORM_CACHE = NORM_CACHE
RUNS = SIZE_NORM_RUNS
CASE = "11_I0000071"          # fold 0, validation
HAVE_DATA = os.path.isdir(CACHE) and os.path.isdir(os.path.join(RUNS, "nnaug_x4norm_s42"))
HAVE_GPU = torch.cuda.is_available()
needs_data = unittest.skipUnless(HAVE_DATA and HAVE_GPU, "needs caches/, the runs and a GPU")


def ckpt(run, seed=42, fold=0):
    return os.path.join(RUNS, f"{run}_s{seed}", f"fold{fold}", "checkpoints", "phase1_last.pth")


def dice(a, b):
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


class Geometry(unittest.TestCase):

    def test_pad_never_crops(self):
        for shape in [(10, 96, 50), (97, 100, 8), (200, 1, 96)]:
            out = pad(torch.zeros(shape), 96)
            for n, t in zip(shape, out.shape):
                self.assertGreaterEqual(t, max(n, 96))
                self.assertEqual(t % 8, 0)

    def test_resample_identity(self):
        x = torch.rand(20, 24, 28)
        aff = np.diag([0.4, 0.4, 0.4, 1.0])
        y = affine_resample(x, aff, aff, x.shape)
        self.assertLess(float((x - y).abs().max()), 1e-5)

    def test_spacing_grid_keeps_extent(self):
        aff = np.diag([0.4, -0.4, 0.4, 1.0])
        out, shape = spacing_grid(aff, (241, 325, 200), 1.6)
        self.assertEqual(shape, (61, 82, 51))
        self.assertEqual(np.sign(out[1, 1]), -1)


@needs_data
class Guard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from common.image_cache import load
        cls.dev = torch.device("cuda")
        cls.pass1 = load_net(ckpt("nnaug_x4e500"), cls.dev)[0]
        cls.norm = load_net(ckpt("nnaug_x4norm"), cls.dev)[0]
        img, cls.aff = load(CACHE, CASE, 1)
        cls.img = np.asarray(img, np.float32).squeeze()

    def test_normal_case_is_normalised(self):
        _, info = segment(self.img, self.aff, self.pass1, [self.norm], 130.0, dev=self.dev)
        self.assertFalse(info["fallback"])
        self.assertTrue(0.5 <= info["k"] <= 2)

    def test_out_of_range_returns_pass1(self):
        # A V0 so large that k < 0.5: the result must be pass 1's own segmentation.
        field, info = segment(self.img, self.aff, self.pass1, [self.norm], 1e5, dev=self.dev)
        self.assertTrue(info["fallback"])
        plain = segment_plain(self.img, self.aff, [self.pass1], dev=self.dev)
        self.assertTrue(torch.equal(field < 0, plain < 0))

    def test_empty_volume(self):
        field, info = segment(np.zeros((120, 130, 110), np.float32), self.aff, self.pass1,
                              [self.norm], 130.0, dev=self.dev)
        self.assertTrue(info["fallback"])
        self.assertEqual(int((field < 0).sum()), 0)
        self.assertFalse(bool(torch.isnan(field).any()))


@needs_data
class Reproduce(unittest.TestCase):
    # The stored evaluation of each run (eval_val/summary.json) against a fresh prediction.

    def scores(self, run):
        with open(os.path.join(RUNS, f"{run}_s42", "fold0", "eval_val", "summary.json")) as f:
            cases = json.load(f)["cases"]
        return next(c for c in cases if c["case"] == CASE)

    def predict(self, *args):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, os.path.join(SIZE_NORM_DIR, "evaluation", "predict.py"), "--case", CASE,
                            "--out-dir", tmp, *args], check=True, stdout=subprocess.DEVNULL)
            from common.image_cache import load_label
            sdf = np.load(os.path.join(tmp, f"{CASE}.npz"))["sdf"]
            lab = np.asarray(load_label(CACHE, CASE)[0]).squeeze().astype(bool)
            return dice(sdf < 0, lab)

    def test_plain_x4(self):
        self.assertAlmostEqual(self.predict("--checkpoint", ckpt("nnaug_x4e500")),
                               self.scores("nnaug_x4e500")["dice_label"], delta=1e-4)

    def test_size_normalised(self):
        # The stored number came from an evaluation cache whose pass-1 volume was measured
        # slightly differently (median 0.1%); the pipeline itself is pinned by the value below.
        d = self.predict("--pass1", ckpt("nnaug_x4e500"), "--checkpoint", ckpt("nnaug_x4norm"))
        self.assertAlmostEqual(d, self.scores("nnaug_x4norm")["dice_label"], delta=2e-3)
        self.assertAlmostEqual(d, REFERENCE_NORM_DICE, delta=1e-5)


@needs_data
class Segment(unittest.TestCase):
    # segment.py on the cached native volume written as NIfTI: in RAS it must give exactly
    # the pipeline's mask; reoriented to LPS, the same mask in that orientation.

    def test_ras_and_lps(self):
        import nibabel as nib
        from common.image_cache import load
        img, aff = load(CACHE, CASE, 1)
        img = np.asarray(img, np.float32).squeeze()
        dev = torch.device("cuda")
        field, _ = segment(img, aff, load_net(ckpt("nnaug_x4e500"), dev)[0],
                           [load_net(ckpt("nnaug_x4norm"), dev)[0]], 130.0, dev=dev)
        expected = (field < 0).cpu().numpy()

        with tempfile.TemporaryDirectory() as tmp:
            ras = nib.Nifti1Image(img, aff)
            # LPS: x and y flipped, the affine flipped with them (the same scan in world space).
            flip = np.diag([-1.0, -1.0, 1.0, 1.0])
            flip[0, 3], flip[1, 3] = img.shape[0] - 1, img.shape[1] - 1
            lps = nib.Nifti1Image(img[::-1, ::-1].copy(), aff @ flip)
            for name, im in (("ras", ras), ("lps", lps)):
                nib.save(im, os.path.join(tmp, f"{name}.nii.gz"))
                subprocess.run([sys.executable, os.path.join(SIZE_NORM_DIR, "deploy", "segment.py"),
                                os.path.join(tmp, f"{name}.nii.gz"), os.path.join(tmp, f"{name}_mask.nii.gz"),
                                "--v0-ml", "130"], check=True, stdout=subprocess.DEVNULL)
            m_ras = np.asarray(nib.load(os.path.join(tmp, "ras_mask.nii.gz")).dataobj)
            m_lps_img = nib.load(os.path.join(tmp, "lps_mask.nii.gz"))
            m_lps = np.asarray(m_lps_img.dataobj)

        self.assertEqual(set(np.unique(m_ras)), {0, 1})
        self.assertTrue(np.array_equal(m_ras.astype(bool), expected))
        self.assertTrue(np.allclose(m_lps_img.affine, aff @ flip))
        self.assertTrue(np.array_equal(m_lps[::-1, ::-1].astype(bool), expected))


@unittest.skipUnless(os.path.isdir(NORM_CACHE), "needs caches/cache_norm4")
class Cache(unittest.TestCase):

    def test_size_norm_rebuild_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "norm")
            links = [os.path.join(dst, "x1"), os.path.join(dst, "label"), os.path.join(dst, "sdf", "x1")]
            try:
                subprocess.run([sys.executable, os.path.join(SIZE_NORM_DIR, "build_cache.py"), "--dst", dst,
                                "--limit", "1", "--v0-ml", "130"], check=True, stdout=subprocess.DEVNULL)
                cid = sorted(os.listdir(os.path.join(dst, "x4")))[0]
                for sub, key in (("x4", "img"), (os.path.join("sdf", "x4"), "sdf"),
                                 (os.path.join("sdf", "x4mm"), "sdf")):
                    with np.load(os.path.join(dst, sub, cid)) as new, \
                         np.load(os.path.join(NORM_CACHE, sub, cid)) as old:
                        self.assertTrue(np.array_equal(new[key], old[key]), sub)
                        self.assertTrue(np.array_equal(new["affine"], old["affine"]), sub)
            finally:
                # size_normalization/build_cache.py links x1/, label/ and sdf/x1/ to caches/cache/ (junctions on
                # Windows). Remove the links themselves first, so cleaning up the temporary
                # folder can never reach into the real cache.
                for link in links:
                    if os.path.lexists(link):
                        os.unlink(link) if os.path.islink(link) else os.rmdir(link)


@unittest.skipUnless(HAVE_DATA and HAVE_GPU, "needs caches/ and a GPU")
class Train(unittest.TestCase):

    def test_two_seeded_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, os.path.join(SIZE_NORM_DIR, "train.py"), "--fold", "1", "--seed", "42",
                            "--epochs", "2", "--run-dir", tmp], check=True, stdout=subprocess.DEVNULL)
            with open(os.path.join(tmp, "train_log.csv")) as f:
                rows = f.read().split("\n")[1:3]
        losses = [float(r.split(",")[1]) for r in rows]
        self.assertEqual(losses, [1.2039, 1.0418])


# The size-normalised Dice of CASE (seed 42, fold 0) through infer.py, recorded when the
# evaluation moved onto it.
REFERENCE_NORM_DICE = 0.9473210834451545


if __name__ == "__main__":
    unittest.main()
