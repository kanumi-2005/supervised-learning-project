import pandas as pd
import argparse
from pathlib import Path
import re

from pandas.core.indexers import is_empty_indexer


# =========================
# LATEX ESCAPE
# =========================
def latex_escape(text):
    if pd.isna(text):
        return ""

    text = str(text)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


# =========================
# LOAD CSV (KEEP RAW STRUCTURE)
# =========================
def load_csv(path, no_header=False):
    df = pd.read_csv(
        path,
        engine="python",
        keep_default_na=False,
        dtype=str
    )

    # giữ cột, chỉ normalize header
    new_cols = []
    for c in df.columns:
        c = str(c).replace("\ufeff", "").strip()

        # Unnamed hoặc rỗng → giữ cột nhưng header trống
        if c.lower().startswith("unnamed") or c == "":
            new_cols.append("")
        else:
            new_cols.append(c)

    df.columns = new_cols

    if no_header:
        df.columns = [str(i) for i in range(df.shape[1])]

    return df


# =========================
# RANGE PARSER
# =========================
def parse_range(token, columns):
    m = re.match(r"^(\d+)-(\d+)$", token)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))

    m = re.match(r"^-(\d+)$", token)
    if m:
        return list(range(0, int(m.group(1)) + 1))

    m = re.match(r"^(\d+)-$", token)
    if m:
        return list(range(int(m.group(1), columns)))

    m = re.match(r"(\d+)$", token)
    if m:
        return list(range(int(m.group(1)), int(m.group(1)) + 1))

    raise ValueError(f"Invalid range: \"{token}\"")


# =========================
# PARSE GROUPS
# =========================
def parse_groups(groups, columns):
    parsed = []

    if groups is None:
        parsed.append(list(range(0, columns)))

    for group in groups:
        cols = []
        tokens = group.split()

        for t in tokens:
            r = parse_range(t, columns)

            if r is not None:
                cols.extend(r)
            elif t.isdigit():
                cols.append(int(t))
            else:
                cols.append(t)

        cols = sorted(set(cols))

        if len(cols) == 0:
            raise ValueError(f"Empty group: \"{group}\"")

        parsed.append(sorted(set(cols)))


    return parsed


# =========================
# RENDER COLUMN NAME
# =========================
def render_colname(c):
    c = str(c)
    if c.strip() == "":
        return ""
    return latex_escape(c)


# =========================
# TABLE GENERATOR
# =========================
def df_to_tblr(df):
    header = " & ".join(render_colname(c) for c in df.columns) + " \\\\"

    rows = [
        " & ".join(latex_escape(v) for v in row.values) + " \\\\"
        for _, row in df.iterrows()
    ]

    colspec = "|".join(["l"] * len(df.columns))

    return f"""\\begin{{tblr}}{{
  width=\\textwidth,
  colspec={{|{colspec}|}},
  row{{1}} = {{font=\\bfseries, bg=gray!15}},
  hlines,
  vlines
}}
{header}
{"\n".join(rows)}
\\end{{tblr}}"""


# =========================
# SAVE FILE
# =========================
def save_tex(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df_to_tblr(df), encoding="utf-8")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("csv_path")
    parser.add_argument("output_prefix")

    parser.add_argument("--groups", nargs="+", default=None)
    parser.add_argument("--shared", nargs=1, default=[])
    parser.add_argument("--include-remaining", action="store_true")
    parser.add_argument("--no-header", action="store_true")

    args = parser.parse_args()


    df = load_csv(args.csv_path, args.no_header)
    n_columns = df.shape[1]

    # =========================
    # SHARED COLUMNS
    # =========================
    parsed  = parse_groups(args.shared, n_columns)
    if len(parsed) == 0:
        shared_group = []
    else:
        shared_group, = parsed

    # =========================
    # GROUP PARSING
    # =========================
    if args.groups is None:
        groups = [[i for i in range(0, n_columns) if i not in shared_group]]
    else:
        parsed = parse_groups(args.groups, n_columns)
        groups = []
        for g in parsed:
            groups.append(g)

    # =========================
    # CHECK DUPLICATE COLUMNS
    # =========================
    all_cols = shared_group + [c for g in groups for c in g]

    if len(all_cols) != len(set(all_cols)):
        raise ValueError("Duplicate columns detected")

    # =========================
    # REMAINING GROUP
    # =========================
    used = set()
    for s in groups:
        used.update(s)
    for g in groups:
        used.update(g)

    if args.include_remaining:
        rest = [c for c in range(n_columns) if c not in used]
        if rest:
            groups.append(rest)

    # =========================
    # EXPORT TABLES
    # =========================
    if len(groups) == 1:
        df_part = df
        out = f"{args.output_prefix}.tex"
        save_tex(df_part, out)
        print("Saved:", out)
        return

    for i, cols in enumerate(groups):
        df_part = df.iloc[:, shared_group + cols]

        out = f"{args.output_prefix}-part{i+1}.tex"
        save_tex(df_part, out)

        print("Saved:", out)


if __name__ == "__main__":
    main()
