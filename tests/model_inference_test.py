import unittest

import numpy as np

from ml_wac.types.attack_type import AttackType
from ml_wac.wac import WebAttackClassifier


class ModelInferenceTest(unittest.TestCase):

    def setUp(self):
        self.model = WebAttackClassifier()

    def test_xss(self):
        result = self.model.predict_single("/test?id=<script>alert(1)</script>")

        self.assertEqual(result, AttackType.XSS)

    def test_lfi(self):
        result = self.model.predict_single("/test?id=/etc/passwd")

        self.assertEqual(result, AttackType.LFI)

    def test_sqli(self):
        result = self.model.predict_single("/endpoint.php?user=1+ORDER+BY+10")

        self.assertEqual(result, AttackType.SQLI)

    def test_rfi(self):
        result = self.model.predict_single("/test?path=https://izak.amsterdam/bad.php")

        self.assertEqual(result, AttackType.RFI)

    def test_batch(self):
        result = self.model.predict([
            "/status?message=<script>/*+Bad+stuff+here+*/</script>",
            "/?download=../include/connection.php",
            "/?file=../../uploads/evil.php",
            "/products?category=Gifts'+OR+1=1--"
        ])

        return self.assertEqual(result.all(), np.array([
            AttackType.XSS,
            AttackType.RFI,
            AttackType.LFI,
            AttackType.SQLI
        ]).all())


if __name__ == '__main__':
    unittest.main()
