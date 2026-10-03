"""The numbers that turn model scores into a risk barrier, fitted by
scripts/calibrate.py on simulated normal shoppers and saved as JSON."""

import bisect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from anomaly.catalog import DATA_DIR

CALIBRATION_PATH = DATA_DIR / "calibration.json"


@dataclass
class Calibration:
    # softmax temperature per mode ("browse", "search"), fitted by maximum likelihood
    tau: dict[str, float] = field(
        default_factory=lambda: {"browse": 0.02, "search": 0.02}
    )
    eps: float = 0.02
    # location and scale of excess surprisal on normal steps, per mode
    loc: dict[str, float] = field(
        default_factory=lambda: {"browse": 0.0, "search": 0.0}
    )
    scale: dict[str, float] = field(
        default_factory=lambda: {"browse": 1.0, "search": 1.0}
    )
    # standardized step scores are clipped to this range before they add up
    clip: tuple[float, float] = (-3.0, 6.0)
    # CUSUM: slack per step, and the barrier it must cross to raise a flag
    cusum_k: float = 0.5
    barrier: float = 5.0
    target_fpr: float = 0.01
    # session z-scores of the normal calibration sessions, sorted
    normal_z: list[float] = field(default_factory=list)
    # how the barrier did on held-out sessions, for the app's "about" panel
    report: dict = field(default_factory=dict)

    def percentile(self, z: float) -> float | None:
        """Share of normal sessions with a lower session z-score."""
        if not self.normal_z:
            return None
        return bisect.bisect_left(self.normal_z, z) / len(self.normal_z)

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "Calibration":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        data["clip"] = tuple(data.get("clip", (-3.0, 6.0)))
        return cls(**data)
