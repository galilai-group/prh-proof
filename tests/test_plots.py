"""Figure functions write their PNGs."""

from pathlib import Path

import pandas as pd
from PIL import Image

from prh import plots


def test_caption_grid(tmp_path):
    paths = []
    for i in range(8):
        paths.append(tmp_path / f"{i}.jpg")
        Image.new("RGB", (80, 60), (30 * i, 0, 0)).save(paths[-1])
    plots.caption_grid(
        paths,
        [["a human caption"]] * 8,
        ["a generated caption that is long enough to wrap"] * 8,
        tmp_path / "grid.png",
        picks=range(8),
    )
    assert (tmp_path / "grid.png").stat().st_size > 0


def test_training_curves(tmp_path):
    run = pd.DataFrame({"epoch": [0, 9, 19], "mknn": [0.01, 0.03, 0.05], "R@1": [0.001, 0.02, 0.04], "R@5": [0.005, 0.06, 0.1]})
    plots.training_curves({"good": (plots.BLUE, [run, run * 1.1])}, tmp_path / "curves.png")
    assert (tmp_path / "curves.png").exists()


def test_davis_kahan(tmp_path):
    plots.davis_kahan(tmp_path / "dk.png", cases=20)
    assert (tmp_path / "dk.png").exists()


def test_paper_figures_render_from_the_shipped_tables(tmp_path):
    import figures

    tables = Path(__file__).resolve().parents[1] / "paper" / "data"
    read = lambda name: pd.read_csv(tables / f"{name}.csv")  # noqa: E731
    figures.scaling(read("scaling"), read("references"), read("image_side"), tmp_path / "scaling.png")
    figures.both_sides(read("msrvtt"), read("msrvtt_dinov2"), tmp_path / "both_sides.png")
    figures.captioner(read("captioner_noise"), read("captioner_pairs"), tmp_path / "captioner.png")
    figures.ablations(read("ablations"), tmp_path / "ablations.png")
    figures.phase(read("phase_diagram"), tmp_path / "phase.png")
    figures.phase_law(read("phase_grid"), read("phase_law_fit"), tmp_path / "phase_law.png")
    figures.teaser(read("phase_real"), tmp_path / "teaser.png")
    names = {"scaling.png", "both_sides.png", "captioner.png", "ablations.png", "phase.png", "phase_law.png", "teaser.png"}
    assert {p.name for p in tmp_path.glob("*.png")} == names


def test_plots_leave_global_matplotlib_settings_alone(tmp_path):
    import matplotlib

    before = dict(matplotlib.rcParams)
    plots.davis_kahan(tmp_path / "dk.png", cases=4)
    assert dict(matplotlib.rcParams) == before
