from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class BuildVolume:
    x_mm: float = 220.0
    y_mm: float = 220.0
    z_mm: float = 250.0


@dataclass(frozen=True)
class ConnectorConfig:
    style: str = "auto"  # auto, hex, dovetail, magnet, none
    tolerance_mm: float = 0.15
    peg_radius_mm: float = 4.0
    peg_depth_mm: float = 8.0
    count: int = 2
    # Specifics
    dovetail_width_mm: float = 10.0
    dovetail_angle_deg: float = 20.0
    magnet_radius_mm: float = 3.05  # Standard 6mm magnet + tolerance
    magnet_depth_mm: float = 3.2    # Standard 3mm magnet + tolerance


@dataclass(frozen=True)
class SplitConfig:
    max_depth: int = 6
    max_pieces: int = 0
    cut_samples: int = 5
    curvature_band_mm: float = 6.0
    roughness_weight: float = 1.0
    density_weight: float = 1.0
    orient_step_deg: int = 90
    cut_margin_mm: float = 2.0
    support_weight: float = 0.5


@dataclass(frozen=True)
class OutputConfig:
    format: str = "3mf"
    engine: str = "manifold"


@dataclass(frozen=True)
class AppConfig:
    build_volume: BuildVolume = BuildVolume()
    connectors: ConnectorConfig = ConnectorConfig()
    split: SplitConfig = SplitConfig()
    output: OutputConfig = OutputConfig()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AppConfig":
        build = data.get("build_volume", {})
        connectors = data.get("connectors", {})
        split = data.get("split", {})
        output = data.get("output", {})
        return AppConfig(
            build_volume=BuildVolume(
                x_mm=float(build.get("x_mm", BuildVolume.x_mm)),
                y_mm=float(build.get("y_mm", BuildVolume.y_mm)),
                z_mm=float(build.get("z_mm", BuildVolume.z_mm)),
            ),
            connectors=ConnectorConfig(
                style=str(connectors.get("style", ConnectorConfig.style)),
                tolerance_mm=float(
                    connectors.get("tolerance_mm", ConnectorConfig.tolerance_mm)
                ),
                peg_radius_mm=float(
                    connectors.get("peg_radius_mm", ConnectorConfig.peg_radius_mm)
                ),
                peg_depth_mm=float(
                    connectors.get("peg_depth_mm", ConnectorConfig.peg_depth_mm)
                ),
                count=int(connectors.get("count", ConnectorConfig.count)),
                dovetail_width_mm=float(
                    connectors.get("dovetail_width_mm", ConnectorConfig.dovetail_width_mm)
                ),
                dovetail_angle_deg=float(
                    connectors.get("dovetail_angle_deg", ConnectorConfig.dovetail_angle_deg)
                ),
                magnet_radius_mm=float(
                    connectors.get("magnet_radius_mm", ConnectorConfig.magnet_radius_mm)
                ),
                magnet_depth_mm=float(
                    connectors.get("magnet_depth_mm", ConnectorConfig.magnet_depth_mm)
                ),
            ),
            split=SplitConfig(
                max_depth=int(split.get("max_depth", SplitConfig.max_depth)),
                max_pieces=int(split.get("max_pieces", SplitConfig.max_pieces)),
                cut_samples=int(split.get("cut_samples", SplitConfig.cut_samples)),
                curvature_band_mm=float(
                    split.get("curvature_band_mm", SplitConfig.curvature_band_mm)
                ),
                roughness_weight=float(
                    split.get("roughness_weight", SplitConfig.roughness_weight)
                ),
                density_weight=float(
                    split.get("density_weight", SplitConfig.density_weight)
                ),
                orient_step_deg=int(
                    split.get("orient_step_deg", SplitConfig.orient_step_deg)
                ),
                cut_margin_mm=float(split.get("cut_margin_mm", SplitConfig.cut_margin_mm)),
                support_weight=float(split.get("support_weight", SplitConfig.support_weight)),
            ),
            output=OutputConfig(
                format=str(output.get("format", OutputConfig.format)),
                engine=str(output.get("engine", OutputConfig.engine)),
            ),
        )


def load_config(path: Optional[Path]) -> AppConfig:
    if not path:
        return AppConfig()
    data = yaml.safe_load(path.read_text()) or {}
    return AppConfig.from_dict(data)
