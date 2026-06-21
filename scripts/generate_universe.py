#!/usr/bin/env python3
import csv
import argparse
from collections import OrderedDict


def detect_encoding_try(path):
    # prefer cp949/euc-kr for Korean CSVs
    for enc in ("cp949", "euc-kr", "utf-8"):
        try:
            with open(path, encoding=enc) as f:
                f.readline()
            return enc
        except Exception:
            continue
    return "utf-8"


def generate(input_path, output_path):
    enc = detect_encoding_try(input_path)
    seen = OrderedDict()
    with open(input_path, newline="", encoding=enc, errors="replace") as inf:
        reader = csv.reader(inf)
        for row in reader:
            if not row:
                continue
            # Expect: code, name, market, ...
            code = row[0].strip().strip('"')
            name = row[1].strip().strip('"') if len(row) > 1 else ""
            market = row[2].strip().strip('"') if len(row) > 2 else ""
            market = market.upper()
            if market not in ("KOSPI", "KOSDAQ"):
                continue
            if code in seen:
                continue
            seen[code] = (name, market)

    with open(output_path, "w", newline="", encoding="utf-8") as outf:
        writer = csv.writer(outf)
        writer.writerow(["종목코드", "종목명", "시장구분"])
        for code, (name, market) in seen.items():
            writer.writerow([code, name, market])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", default="data/universe.csv")
    args = p.parse_args()
    generate(args.input, args.output)


if __name__ == "__main__":
    main()
