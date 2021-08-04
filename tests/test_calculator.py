from torch.testing import assert_allclose
from pytest import mark
from glob import glob
from os.path import dirname, join
from torchmdnet.calculators import External
from torchmdnet.models.model import load_model

from utils import create_example_batch


@mark.parametrize(
    "checkpoint",
    glob(
        join(dirname(dirname(__file__)), "examples", "pretrained", "**", "*.ckpt"),
        recursive=True,
    ),
)
def test_load_model(checkpoint):
    z, pos, _ = create_example_batch(multiple_batches=False)
    calc = External(checkpoint, z.unsqueeze(0))
    model = load_model(checkpoint, derivative=True)

    e_calc, f_calc = calc.calculate(pos, None)
    e_pred, f_pred = model(z, pos)

    assert_allclose(e_calc, e_pred)
    assert_allclose(f_calc, f_pred.unsqueeze(0))