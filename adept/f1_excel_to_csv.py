#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the F1 Excel workbook to the CSV validate_f1_families.py expects,
computing age_days as (sampling date - mating date) from the Excel date serials.

    python f1_excel_to_csv.py --xlsx <F1 workbook>.xlsx \
        --out f1_data.csv
"""
import argparse, openpyxl, csv

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", required=True)
ap.add_argument("--sheet", default=None)
ap.add_argument("--out", default="f1_data.csv")
a = ap.parse_args()

wb = openpyxl.load_workbook(a.xlsx, data_only=True)
ws = wb[a.sheet] if a.sheet else wb[wb.sheetnames[0]]
hdr = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]

# The two date columns carry their original Chinese headers in the source
# workbook; they are escaped below as \u7e41\u6b96\u65f6\u95f4 (mating date)
# and \u53d6\u6837\u65f6\u95f4 (sampling date).
# The two date columns carry their original Chinese headers in the source
# workbook; they are escaped below as \u7e41\u6b96\u65f6\u95f4 (mating date)
# and \u53d6\u6837\u65f6\u95f4 (sampling date).
need = ("group", "M", "F", "\u7e41\u6b96\u65f6\u95f4", "\u53d6\u6837\u65f6\u95f4")
missing = [x for x in need if x not in hdr]
if missing:
    raise SystemExit(f"Excel missing expected columns {missing}; found {hdr}")

bi, si = hdr.index("\u7e41\u6b96\u65f6\u95f4"), hdr.index("\u53d6\u6837\u65f6\u95f4")
with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr + ["age_days"])
    for r in range(2, ws.max_row + 1):
        row = [ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
        if row[0] is None:
            continue
        b, s = row[bi], row[si]
        age = (s - b) if isinstance(b, (int, float)) and isinstance(s, (int, float)) else ""
        w.writerow(row + [age])
print(f"-> {a.out}")
