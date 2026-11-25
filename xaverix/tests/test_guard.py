"""Grounding guard: reject unfaithful output, accept faithful rewrites.

Each rejection test is a regression fixture for a class of model error
observed during evaluation: a fabricated pathogen, an invented dose for a real
product, a plausible-looking but non-existent fungicide, and unsupported
promotional claims. The acceptance test guards the opposite failure, a guard
so strict that valid paraphrase never passes.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import guard, knowledge  # noqa: E402

CONTEXT = {
    "crop": "tomato",
    "region": "tamil_nadu",
    "region_label": "Tamil Nadu",
    "severity": 38,
    "band": "moderate",
    "confidence_pct": 74,
}

FAITHFUL = """WHAT IS HAPPENING:
Your tomato leaf shows brown dead patches with yellowing around them. About 38 percent of the leaf blade is damaged.
WHAT TO DO NOW:
Act within 24 hours. Remove the affected lower leaves and check the neighbouring plants, because the damage is spreading. Spray mancozeb 75 WP at 2 g per litre of water, every 10 days.
ORGANIC OPTION:
Spray neem oil at 5 ml per litre of water every 7 days, in the evening, wetting both sides of the leaf.
PREVENT NEXT SEASON:
Switch from overhead watering to drip, mulch the soil, and rotate to a non-host crop for two to three seasons."""


def _wrap(what, now, organic, prevent):
    return (
        "WHAT IS HAPPENING:\n%s\nWHAT TO DO NOW:\n%s\n"
        "ORGANIC OPTION:\n%s\nPREVENT NEXT SEASON:\n%s" % (what, now, organic, prevent)
    )


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.infected = knowledge.entry_for("infected")
        self.healthy = knowledge.entry_for("healthy")

    # ------------------------------------------------------ must accept
    def test_accepts_a_faithful_rewrite(self):
        result = guard.check(FAITHFUL, self.infected, CONTEXT)
        self.assertTrue(result["ok"], result["violations"])
        self.assertEqual(result["decision"], "accept")
        self.assertGreater(result["support"], 0.70)

    # ------------------------------------------------------ must reject
    def test_rejects_a_fabricated_pathogen(self):
        text = _wrap(
            "Early blight of tomato is caused by the bacterium Colletobacter, "
            "which spreads through the air.",
            "Remove the affected lower leaves today.",
            "Spray neem oil at 5 ml per litre of water every 7 days.",
            "Rotate to a non-host crop for two to three seasons.",
        )
        result = guard.check(text, self.infected, CONTEXT)
        self.assertFalse(result["ok"])
        self.assertIn("ORGANISM_NOT_IN_KB", result["codes"])
        self.assertTrue(
            any("Colletobacter" in v["detail"] for v in result["violations"]),
            result["violations"],
        )

    def test_rejects_an_invented_dose_for_a_real_product(self):
        text = _wrap(
            "The leaf shows brown spots with rings.",
            "Spray mancozeb 75 WP at 9 g per litre of water every 10 days.",
            "Spray neem oil at 5 ml per litre of water every 7 days.",
            "Remove the affected lower leaves and rotate the crop.",
        )
        result = guard.check(text, self.infected, CONTEXT)
        self.assertFalse(result["ok"])
        self.assertIn("NUMBER_NOT_IN_KB", result["codes"])

    def test_rejects_a_product_not_in_the_knowledge_base(self):
        text = _wrap(
            "The leaf shows brown dead patches.",
            "Spray azoxystrobin at 2 g per litre of water every 10 days.",
            "Spray neem oil at 5 ml per litre of water every 7 days.",
            "Remove the affected lower leaves and rotate the crop.",
        )
        result = guard.check(text, self.infected, CONTEXT)
        self.assertFalse(result["ok"])
        self.assertIn("CHEMICAL_NOT_IN_KB", result["codes"])

    def test_rejects_an_invented_plausible_sounding_fungicide(self):
        # Not on the lexicon, but it has fungicide morphology, so the suffix
        # rule must catch it.
        text = _wrap(
            "The leaf shows brown dead patches.",
            "Spray zanoconazole at 2 g per litre of water every 10 days.",
            "Spray neem oil at 5 ml per litre of water every 7 days.",
            "Remove the affected lower leaves and rotate the crop.",
        )
        result = guard.check(text, self.infected, CONTEXT)
        self.assertFalse(result["ok"])
        self.assertIn("CHEMICAL_NOT_IN_KB", result["codes"])

    def test_rejects_guarantees_and_prices(self):
        text = _wrap(
            "The leaf shows brown dead patches.",
            "Buy the treatment for Rs 450 and it will cure the plant, guaranteed.",
            "Spray neem oil at 5 ml per litre of water every 7 days.",
            "Remove the affected lower leaves and rotate the crop.",
        )
        result = guard.check(text, self.infected, CONTEXT)
        self.assertFalse(result["ok"])
        self.assertIn("BANNED_PATTERN", result["codes"])

    def test_rejects_a_disease_claim_on_a_healthy_leaf(self):
        # The healthy entry licenses no pathogen and no kingdom at all.
        text = _wrap(
            "This leaf is infected by a fungus called Alternaria solani.",
            "No treatment is needed today.",
            "Spray neem oil at 3 ml per litre of water once every 15 days.",
            "Water at the base with drip and remove crop residue after harvest.",
        )
        result = guard.check(text, self.healthy, CONTEXT)
        self.assertFalse(result["ok"])
        self.assertTrue(
            {"ORGANISM_NOT_IN_KB", "PATHOGEN_TYPE_MISMATCH"} & set(result["codes"]),
            result["codes"],
        )

    def test_rejects_missing_headings(self):
        result = guard.check(
            "Your leaf is sick. Spray neem oil at 5 ml per litre of water.",
            self.infected,
            CONTEXT,
        )
        self.assertFalse(result["ok"])
        self.assertIn("FORMAT", result["codes"])

    def test_rejects_empty_output(self):
        result = guard.check("", self.infected, CONTEXT)
        self.assertFalse(result["ok"])
        self.assertEqual(result["codes"], ["EMPTY"])

    def test_rejects_another_region(self):
        text = FAITHFUL.replace(
            "Switch from overhead watering",
            "In Punjab, switch from overhead watering",
        )
        result = guard.check(text, self.infected, CONTEXT)
        self.assertFalse(result["ok"])
        self.assertIn("WRONG_REGION", result["codes"])

    # --------------------------------------------------- number handling
    def test_list_numbering_is_not_treated_as_a_claim(self):
        text = _wrap(
            "The leaf shows brown dead patches with yellowing around them.",
            "1. Remove the affected lower leaves today.\n"
            "2. Spray mancozeb 75 WP at 2 g per litre of water every 10 days.",
            "Spray neem oil at 5 ml per litre of water every 7 days.",
            "Mulch the soil and rotate to a non-host crop for two to three seasons.",
        )
        result = guard.check(text, self.infected, CONTEXT)
        self.assertNotIn("NUMBER_NOT_IN_KB", result["codes"], result["violations"])

    def test_devanagari_digits_are_normalised(self):
        self.assertEqual(guard.normalise_digits("२ ग्राम"), "2 ग्राम")
        self.assertEqual(guard.normalise_digits("௫ மில்லி"), "5 மில்லி")

    # ------------------------------------------------------------- demo
    def test_demo_rejects_every_fixture(self):
        report = guard.demo()
        self.assertTrue(report["all_rejected"])
        self.assertEqual(len(report["cases"]), 3)
        for case in report["cases"]:
            self.assertEqual(case["decision"], "reject")
            self.assertTrue(case["codes"])


if __name__ == "__main__":
    unittest.main()
