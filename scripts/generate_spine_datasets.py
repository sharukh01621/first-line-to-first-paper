#!/usr/bin/env python3
"""Generate realistic, seeded Public Health & Epidemiology spine datasets.

Writes identical CSV copies into:
  1. colab_drive/datasets/
  2. sharukh_tutor/src/sharukh_tutor/data/
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIRS = [
    ROOT / "colab_drive" / "datasets",
    ROOT / "sharukh_tutor" / "src" / "sharukh_tutor" / "data",
]


def _write_csv(filename: str, fieldnames: list[str], rows: list[dict]) -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def generate_clinic_triage(seed: int = 42, n: int = 120) -> None:
    rng = random.Random(seed)
    districts = ["Dhaka", "Sylhet", "Rajshahi", "Khulna", "Chattogram", "Rangpur"]
    rows = []
    for i in range(1, n + 1):
        age = rng.randint(1, 78)
        sex = rng.choice(["F", "M"])
        if age < 12:
            height_cm = round(rng.uniform(75, 145), 1)
            weight_kg = round(rng.uniform(9, 38), 1)
        else:
            height_cm = round(rng.gauss(162 if sex == "M" else 153, 7.5), 1)
            height_cm = max(138.0, min(190.0, height_cm))
            bmi_target = max(15.5, min(35.0, rng.gauss(22.4, 3.8)))
            weight_kg = round(bmi_target * ((height_cm / 100.0) ** 2), 1)

        temp_c = round(max(36.1, min(40.8, rng.gauss(37.4, 0.95))), 1)
        systolic_bp = int(max(85, min(195, round(rng.gauss(112 + age * 0.42, 13)))))
        hb = round(max(7.2, min(16.8, rng.gauss(12.2 if sex == "F" else 13.8, 1.6))), 1)
        fever_days = 0 if temp_c < 37.8 else rng.randint(1, 9)

        if temp_c >= 39.5 or systolic_bp >= 170 or hb < 8.5:
            priority = "Urgent"
        elif temp_c >= 38.0 or systolic_bp >= 140 or hb < 11.0:
            priority = "Moderate"
        else:
            priority = "Routine"

        rows.append(
            {
                "patient_id": f"UHC-{i:04d}",
                "district": rng.choice(districts),
                "age_years": age,
                "sex": sex,
                "weight_kg": weight_kg,
                "height_cm": height_cm,
                "temp_c": temp_c,
                "systolic_bp": systolic_bp,
                "hemoglobin_g_dl": hb,
                "fever_days": fever_days,
                "triage_priority": priority,
            }
        )

    _write_csv(
        "clinic_triage.csv",
        list(rows[0].keys()),
        rows,
    )
    print(f"Generated clinic_triage.csv ({len(rows)} rows)")


def generate_mch_survey(seed: int = 101, n: int = 500) -> None:
    rng = random.Random(seed)
    divisions = ["Dhaka", "Chattogram", "Sylhet", "Rajshahi", "Khulna", "Barishal", "Rangpur", "Mymensingh"]
    quintiles = ["Poorest", "Poorer", "Middle", "Richer", "Richest"]
    rows = []
    for i in range(1, n + 1):
        residence = rng.choices(["Rural", "Urban"], weights=[0.68, 0.32])[0]
        q_idx = rng.choices([0, 1, 2, 3, 4], weights=[0.26, 0.24, 0.20, 0.16, 0.14] if residence == "Rural" else [0.10, 0.14, 0.20, 0.26, 0.30])[0]
        wealth = quintiles[q_idx]
        maternal_age = rng.randint(17, 43)
        anc_base = 2.2 + 0.7 * q_idx + (0.9 if residence == "Urban" else 0.0)
        anc_visits = max(0, min(10, int(round(rng.gauss(anc_base, 1.7)))))

        # Realistic DHS-style missingness codes (999 or blank) for Part IV cleaning chapters
        if rng.random() < 0.03:
            anc_visits_val: object = 999
        else:
            anc_visits_val = anc_visits

        haz = round(rng.gauss(-1.35 + 0.28 * q_idx, 1.15), 2)
        stunted = 1 if haz < -2.0 else 0
        hb = round(max(7.5, min(15.2, rng.gauss(11.1 + 0.22 * q_idx, 1.35))), 1)
        anemic = 1 if hb < 11.0 else 0
        vaccinated = 1 if rng.random() < (0.72 + 0.05 * q_idx) else 0

        # Occasional realistic messy casing/missingness for Chapter 18 cleaning
        res_str = residence.lower() if rng.random() < 0.02 else residence
        hb_val: object = "" if rng.random() < 0.025 else hb

        rows.append(
            {
                "household_id": f"MCH-{i:05d}",
                "cluster_id": rng.randint(1, 50),
                "division": rng.choice(divisions),
                "residence": res_str,
                "wealth_quintile": wealth,
                "maternal_age_years": maternal_age,
                "anc_visits": anc_visits_val,
                "maternal_hb_g_dl": hb_val,
                "maternal_anemia": anemic,
                "child_haz": haz,
                "child_stunted": stunted,
                "fully_vaccinated": vaccinated,
            }
        )

    _write_csv("mch_survey_sample.csv", list(rows[0].keys()), rows)
    print(f"Generated mch_survey_sample.csv ({len(rows)} rows)")


def generate_epi_surveillance(seed: int = 2026) -> None:
    rng = random.Random(seed)
    districts = [
        ("Dhaka", 950000),
        ("Chattogram", 520000),
        ("Sylhet", 310000),
        ("Rajshahi", 280000),
    ]
    rows = []
    for district, pop in districts:
        scale = pop / 300000.0
        for week in range(1, 105):  # 2 full years of weekly surveillance
            year = 2024 if week <= 52 else 2025
            iso_week = week if week <= 52 else week - 52
            # Monsoon seasonality peaking around weeks 26-38
            monsoon = max(0.0, math.sin((iso_week - 16) * math.pi / 26.0))
            rainfall_mm = round(max(0.0, rng.gauss(18 + 185 * monsoon, 32)), 1)
            mean_temp_c = round(21.5 + 8.5 * math.sin((iso_week - 8) * math.pi / 26.0) + rng.gauss(0, 1.1), 1)
            expected_cases = (14 + 95 * (monsoon ** 1.4)) * scale
            dengue_cases = max(0, int(round(rng.gauss(expected_cases, math.sqrt(expected_cases) * 1.8))))
            resp_cases = max(5, int(round(rng.gauss((55 + 40 * (1 - monsoon)) * scale, 12))))

            rows.append(
                {
                    "year": year,
                    "epi_week": iso_week,
                    "district": district,
                    "population": pop,
                    "rainfall_mm": rainfall_mm,
                    "mean_temp_c": mean_temp_c,
                    "dengue_cases": dengue_cases,
                    "respiratory_cases": resp_cases,
                }
            )

    _write_csv("epi_surveillance.csv", list(rows[0].keys()), rows)
    print(f"Generated epi_surveillance.csv ({len(rows)} rows)")


if __name__ == "__main__":
    generate_clinic_triage()
    generate_mch_survey()
    generate_epi_surveillance()
