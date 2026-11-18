from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from training import fit_weight


class WeightTests(unittest.TestCase):
    def test_selects_by_held_out_kg(self):
        lengths = [202, 230, 190, 250, 162, 233, 202, 240]
        widths = [89, 77, 85, 78, 70, 75, 80, 66]
        weights = [252, 344, 457, 379, 381, 324, 308, 421]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "sessao.csv"
            path.write_text("peso_kg,comprimento_cm,largura_cm\n" + "".join(
                f"{p},{c},{l}\n" for p, c, l in zip(weights, lengths, widths)
            ))
            output = StringIO()
            with patch("sys.argv", ["fit_weight", str(path)]), redirect_stdout(output):
                fit_weight.main()
            self.assertIn("Melhor preditor: largura (erro absoluto medio LOO: 60.2 kg)", output.getvalue())

    def test_same_tag_is_one_animal_and_unweighed_rows_skip(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "sessao.csv"
            path.write_text("peso_kg,comprimento_cm,largura_cm,brinco\n"
                            "300,200,70,A\n300,210,74,A\n,180,60,B\n")
            self.assertEqual(fit_weight.carregar(path), [{"peso": 300, "comp": 205, "larg": 72}])

    def test_degenerate_fold_is_ineligible(self):
        error, _ = fit_weight.validacao_cruzada([1, 1, 2], [10, 20, 30])
        self.assertEqual(error, float("inf"))
