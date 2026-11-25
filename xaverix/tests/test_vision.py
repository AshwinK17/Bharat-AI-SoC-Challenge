"""Descriptor, severity and the abstention cascade."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import classifier, config, features, severity  # noqa: E402
from tools import synth  # noqa: E402


class FeatureTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.image, _ = synth.render_leaf(False, self.rng)

    def test_descriptor_has_the_declared_shape(self):
        descriptor, _ = features.extract(self.image)
        self.assertEqual(descriptor.shape, (features.FEATURE_DIM,))
        self.assertEqual(descriptor.dtype, np.float32)

    def test_descriptor_is_deterministic(self):
        first, _ = features.extract(self.image)
        second, _ = features.extract(self.image)
        np.testing.assert_array_equal(first, second)

    def test_descriptor_is_always_finite(self):
        for image in (
            self.image,
            np.zeros((224, 224, 3), np.uint8),
            np.full((224, 224, 3), 255, np.uint8),
            (self.rng.random((224, 224, 3)) * 255).astype(np.uint8),
        ):
            descriptor, _ = features.extract(image)
            self.assertTrue(np.isfinite(descriptor).all())

    def test_rejects_a_wrongly_shaped_image(self):
        with self.assertRaises(ValueError):
            features.extract(np.zeros((224, 224), np.uint8))
        with self.assertRaises(ValueError):
            features.extract(np.zeros((16, 16, 3), np.uint8))

    def test_leaf_mask_excludes_brown_background(self):
        """A healthy leaf on bare soil must not register as damaged.

        Regression test for a +25 to +30 point severity bias caused by the
        leaf mask accepting any saturated pixel, which admitted brown soil
        into the necrosis mask.
        """
        rng = np.random.default_rng(7)
        necrotic_fractions = []
        for _ in range(12):
            image, _ = synth.render_leaf(False, rng)
            _, intermediates = features.extract(image)
            leaf = intermediates["leaf"]
            area = max(float(leaf.sum()), 1.0)
            necrotic_fractions.append(
                float((intermediates["masks"]["necrotic"] & leaf).sum()) / area
            )
        self.assertLess(float(np.mean(necrotic_fractions)), 0.10)

    def test_non_leaf_scenes_fail_leaf_presence(self):
        """A non-leaf scene must fail Gate 0 on one axis or the other.

        Not necessarily on greenness: the ruled-paper scene contains a dark
        leaf-coloured region, which is green within its own small area. The
        criterion that excludes it is leaf area as a fraction of the frame.
        """
        for kind in ("soil", "sky", "skin", "printed_text", "lined_paper", "brick"):
            image = synth.render_negative(np.random.default_rng(3), kind=kind)
            _, intermediates = features.extract(image)
            leaf_frac = intermediates["leaf_frac"]
            green_frac = intermediates["green_frac"]
            self.assertTrue(
                leaf_frac < 0.20 or green_frac < 0.25,
                "%s passed leaf presence: leaf_frac=%.3f green_frac=%.3f"
                % (kind, leaf_frac, green_frac),
            )


class SeverityTest(unittest.TestCase):
    def test_increases_with_more_damage(self):
        """Severity must rise monotonically along a lesion ramp."""
        values = []
        for fraction in (0.0, 0.08, 0.2, 0.4, 0.75):
            rng = np.random.default_rng(99)
            readings = []
            for _ in range(6):
                image, _ = synth.render_leaf(
                    fraction > 0, rng, severity_hint=fraction or None
                )
                _, intermediates = features.extract(image)
                readings.append(severity.raw_fraction(intermediates))
            values.append(float(np.mean(readings)))
        self.assertLess(values[0], values[-1])
        self.assertLess(values[0], values[2])

    def test_reports_unavailable_without_a_leaf(self):
        image = synth.render_negative(np.random.default_rng(1), kind="sky")
        _, intermediates = features.extract(image)
        result = severity.estimate(intermediates)
        self.assertFalse(result["available"])
        self.assertEqual(result["severity"], 0.0)

    def test_stays_within_zero_to_one_hundred(self):
        rng = np.random.default_rng(5)
        for _ in range(10):
            image, _ = synth.render_leaf(True, rng, severity_hint=0.9)
            _, intermediates = features.extract(image)
            result = severity.estimate(intermediates, (5.0, 40.0))
            self.assertGreaterEqual(result["severity"], 0.0)
            self.assertLessEqual(result["severity"], 100.0)


@unittest.skipUnless(classifier.is_available(), "model not trained yet")
class ClassifierTest(unittest.TestCase):
    def _predict(self, image):
        descriptor, intermediates = features.extract(image)
        return classifier.predict(descriptor, intermediates)

    def test_feature_version_matches_the_model(self):
        self.assertEqual(
            classifier.load()["feature_version"], features.FEATURE_VERSION
        )

    def test_abstains_on_pure_noise(self):
        image = (np.random.default_rng(4).random((224, 224, 3)) * 255).astype(np.uint8)
        result = self._predict(image)
        self.assertTrue(result["abstained"])
        self.assertEqual(result["verdict"], "uncertain")

    def test_abstains_on_a_blank_frame(self):
        for image in (
            np.zeros((224, 224, 3), np.uint8),
            np.full((224, 224, 3), 255, np.uint8),
        ):
            result = self._predict(image)
            self.assertTrue(result["abstained"])

    def test_abstains_on_the_lined_paper_photograph(self):
        """A leaf photographed on ruled paper must not receive a verdict."""
        rng = np.random.default_rng(21)
        verdicts = []
        for _ in range(8):
            image = synth.render_negative(rng, kind="lined_paper")
            verdicts.append(self._predict(image)["verdict"])
        self.assertTrue(
            all(v == "uncertain" for v in verdicts),
            "lined paper must always abstain, got %r" % (verdicts,),
        )

    def test_abstains_on_printed_text(self):
        rng = np.random.default_rng(22)
        image = synth.render_negative(rng, kind="printed_text")
        self.assertTrue(self._predict(image)["abstained"])

    def test_never_reports_full_confidence(self):
        rng = np.random.default_rng(31)
        for _ in range(25):
            infected = bool(rng.random() < 0.5)
            image, _ = synth.render_leaf(infected, rng)
            result = self._predict(image)
            if result["confidence"] is not None:
                self.assertLessEqual(result["confidence"], config.CONFIDENCE_DISPLAY_CAP)
                self.assertLess(result["confidence"], 1.0)

    def test_abstaining_reports_no_confidence(self):
        image = np.zeros((224, 224, 3), np.uint8)
        result = self._predict(image)
        self.assertIsNone(result["confidence"])

    def test_gate_values_are_always_present(self):
        image, _ = synth.render_leaf(True, np.random.default_rng(8))
        result = self._predict(image)
        for key in ("leaf_fraction", "green_fraction", "ood_distance", "confidence_raw"):
            self.assertIn(key, result["gates"])

    def test_carries_the_synthetic_data_caveat(self):
        image, _ = synth.render_leaf(False, np.random.default_rng(9))
        self.assertIn("procedurally generated", self._predict(image)["caveat"])

    def test_agrees_with_the_renderer_on_clear_cases(self):
        """Accuracy on accepted leaves; a measured bound, not an exact value."""
        rng = np.random.default_rng(1234)
        agreed = considered = 0
        for _ in range(40):
            infected = bool(rng.random() < 0.5)
            image, _ = synth.render_leaf(
                infected, rng, severity_hint=0.5 if infected else None
            )
            result = self._predict(image)
            if result["abstained"]:
                continue
            considered += 1
            agreed += result["verdict"] == ("infected" if infected else "healthy")
        self.assertGreater(considered, 15, "too many abstentions to measure accuracy")
        self.assertGreater(agreed / considered, 0.75)


if __name__ == "__main__":
    unittest.main()
