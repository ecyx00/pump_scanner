# strategies/v1_volume_oi_spike/config.py
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    nz: int
    np_: int
    w_vol: float
    w_oi: float
    w_price: float
    trigger_percentile: int
    cooldown_bars: int
    symbols_csv: str

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        return Config(
            nz=int(cfg["nz"]),
            np_=int(cfg["np"]),
            w_vol=float(cfg["w_vol"]),
            w_oi=float(cfg["w_oi"]),
            w_price=float(cfg["w_price"]),
            trigger_percentile=int(cfg["trigger_percentile"]),
            cooldown_bars=int(cfg["cooldown_bars"]),
            symbols_csv=str(cfg["symbols_csv"]),
        )
