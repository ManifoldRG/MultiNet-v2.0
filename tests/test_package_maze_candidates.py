from analysis.package_maze_candidates import (
    DEFAULT_CANDIDATE_DIR,
    DEFAULT_RENDER_DIR,
    package_candidate,
)


def test_package_candidate_builds_illustrated_walkthrough(tmp_path):
    result = package_candidate(
        DEFAULT_CANDIDATE_DIR / "balanced_01.json",
        DEFAULT_CANDIDATE_DIR,
        tmp_path,
        DEFAULT_RENDER_DIR,
    )
    set_dir = result["set_dir"]
    images = sorted((set_dir / "images").glob("*.png"))
    summary = (set_dir / "SUMMARY.md").read_text()

    assert result["image_count"] == 50
    assert len(images) == 50
    assert images[0].name.startswith("001_")
    assert images[-1].name.startswith("050_")
    assert "## Controlled comparisons" in summary
    assert "## Long-tail allocation" in summary
    assert "## Illustrated maze-by-maze walkthrough" in summary
    assert summary.count("maze render](images/") == 50
    assert (set_dir / "candidate_curve.png").exists()
    assert (set_dir / "candidate.csv").exists()
    assert (set_dir / "candidate.json").exists()
