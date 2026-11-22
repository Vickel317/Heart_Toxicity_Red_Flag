# Pull the compounds reported for a short list of Indonesian medicinal plants
# out of the COCONUT 2.0 bulk download.
#
# The bulk CSV is about 700 MB, so it is streamed in chunks straight out of the
# zip rather than loaded whole. Reading it with plain pandas.read_csv on a
# laptop is what makes this step feel impossible; chunked, it is a few minutes.
#
# Species naming is the real difficulty here, not the file size. COCONUT's
# `organisms` field is a free text, pipe separated list contributed from many
# source databases, so one plant appears under its accepted name, its synonyms,
# and several spellings. Matching on the exact binomial alone silently loses
# most of the rows, and a silent loss here looks like "this plant has no known
# compounds", which is the wrong answer.

import glob
import os
import re
import zipfile

import pandas as pd

# Where the bulk download might reasonably be sitting, relative to a notebook
# in notebooks/. COCONUT reissues the file monthly and the release date is in
# the name (coconut_csv-11-2026.csv for November 2026), so nothing here pins a
# single filename: a pinned name breaks silently the moment a newer release is
# downloaded. The file may be left zipped or unzipped, and either works.
BULK_SEARCH_DIRS = [
    os.path.join("..", "data", "coconut"),
    os.path.join("..", "data", "raw"),
    os.path.join("data", "coconut"),
    os.path.join("data", "raw"),
]

BULK_PATTERNS = ["coconut_csv*.zip", "coconut_csv*.csv"]

DOWNLOAD_HINT = (
    "Download the COCONUT CSV bulk file from "
    "https://coconut.naturalproducts.net/download and save it under "
    "data/raw/ (the zip or the unzipped csv, either is fine)."
)


def find_bulk(search_dirs=None):
    # Locate the COCONUT bulk file without hard coding a release name.
    # Raises with the download instructions rather than returning None, because
    # a missing dataset should stop the notebook loudly, not produce an empty
    # result that reads like "these plants have no compounds".
    search_dirs = search_dirs or BULK_SEARCH_DIRS

    found = []
    for directory in search_dirs:
        for pattern in BULK_PATTERNS:
            found.extend(glob.glob(os.path.join(directory, pattern)))

    if not found:
        raise FileNotFoundError("No COCONUT bulk file found. " + DOWNLOAD_HINT)

    # Sorting by name puts the newest release last, because the dates are in
    # the filenames. Preferring the unzipped csv when both exist just avoids
    # paying for decompression on every chunk.
    found.sort()
    plain = [f for f in found if f.lower().endswith(".csv")]
    return (plain or found)[-1]


def _bulk_reader(path, member, chunksize, usecols):
    # Open the bulk file for chunked reading, whether it is a zip or a plain
    # csv, and return (chunk iterator, cleanup callable).
    read_kwargs = dict(
        chunksize=chunksize,
        usecols=usecols,
        dtype=str,
        on_bad_lines="warn",
        low_memory=False,
    )

    if str(path).lower().endswith(".zip"):
        archive = zipfile.ZipFile(path)
        if member is None:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if len(names) != 1:
                archive.close()
                raise ValueError(f"expected one CSV in the zip, found {names}")
            member = names[0]
        handle = archive.open(member)

        def close():
            handle.close()
            archive.close()

        return pd.read_csv(handle, **read_kwargs), close

    # Plain csv on disk.
    reader = pd.read_csv(path, **read_kwargs)
    return reader, lambda: None

# The jamu species, each with the name variants actually seen in natural
# product databases. Every variant maps back to one canonical label so the
# output can be grouped by plant.
#
# These are synonyms and spelling variants of the SAME plant, not related
# species. Phyllanthus amarus is deliberately NOT folded into P. niruri:
# the two are frequently confused in the literature but they are different
# species, so merging them would overstate what the data says.
JAMU_SPECIES = {
    "Curcuma longa": [
        "curcuma longa",
        "curcuma domestica",       # accepted synonym
        "curcuma domestica valeton",
    ],
    "Curcuma xanthorrhiza": [
        "curcuma xanthorrhiza",
        "curcuma zanthorrhiza",    # very common alternative spelling
        "curcuma xanthorriza",
    ],
    "Zingiber officinale": [
        "zingiber officinale",
    ],
    "Andrographis paniculata": [
        "andrographis paniculata",
        "justicia paniculata",     # older basionym
    ],
    "Phyllanthus niruri": [
        "phyllanthus niruri",
    ],
    "Orthosiphon aristatus": [
        "orthosiphon aristatus",
        "orthosiphon stamineus",   # accepted synonym, far more common in papers
        "orthosiphon spicatus",
    ],
}

# Columns kept from the bulk file. The rest of COCONUT's 44 columns are not
# needed downstream and dropping them early keeps the chunks small.
KEEP_COLUMNS = [
    "identifier",
    "canonical_smiles",
    "name",
    "molecular_weight",
    "alogp",
    "np_classifier_pathway",
    "np_classifier_superclass",
    "np_classifier_class",
    "organisms",
]

_SPACES = re.compile(r"\s+")


def _build_patterns(species_map):
    # One compiled word-boundary regex per variant.
    #
    # Word boundaries matter: a bare substring test for "curcuma longa" would
    # also hit "Curcuma longaeflora" if such a name appeared. \b after the
    # final token prevents that, while still allowing the authority suffixes
    # that databases append, as in "Curcuma longa L." or "... Roscoe".
    patterns = []
    for canonical, variants in species_map.items():
        for variant in variants:
            pattern = re.compile(r"\b" + re.escape(variant) + r"\b")
            patterns.append((canonical, variant, pattern))
    return patterns


def _normalise_organism(text):
    return _SPACES.sub(" ", str(text).lower()).strip()


def match_species(organisms_field, patterns):
    # Which of our species does one row's organisms field mention?
    # Returns a sorted list of canonical species names, empty if none.
    if organisms_field is None or pd.isna(organisms_field):
        return []
    text = _normalise_organism(organisms_field)
    if not text:
        return []
    hits = set()
    for canonical, _variant, pattern in patterns:
        if pattern.search(text):
            hits.add(canonical)
    return sorted(hits)


def filter_bulk(
    zip_path,
    member=None,
    species_map=None,
    chunksize=50000,
    progress_every=20,
):
    # Stream the COCONUT bulk CSV and keep only rows mentioning our species.
    # Returns (filtered DataFrame, stats dict).
    species_map = species_map or JAMU_SPECIES
    patterns = _build_patterns(species_map)

    # zip_path may be a zip or an already unzipped csv; _bulk_reader handles both.
    reader, close = _bulk_reader(
        zip_path, member, chunksize, lambda c: c in KEEP_COLUMNS
    )

    kept_chunks = []
    rows_seen = 0
    chunk_index = 0

    try:
        for chunk in reader:
            chunk_index += 1
            rows_seen += len(chunk)

            # Cheap prefilter before the regex work. Most of COCONUT has
            # nothing to do with these six plants, so a plain substring
            # test on the genus throws out the bulk of each chunk first.
            genera = [v.split(" ")[0] for _c, v, _p in patterns]
            genus_re = "|".join(sorted(set(re.escape(g) for g in genera)))
            maybe = chunk["organisms"].fillna("").str.contains(
                genus_re, case=False, regex=True
            )
            candidate = chunk[maybe]

            if len(candidate):
                matched = candidate["organisms"].apply(
                    lambda f: match_species(f, patterns)
                )
                hit = candidate[matched.str.len() > 0].copy()
                if len(hit):
                    hit["matched_species"] = matched[matched.str.len() > 0].apply(
                        lambda s: "|".join(s)
                    )
                    kept_chunks.append(hit)

            if progress_every and chunk_index % progress_every == 0:
                found = sum(len(c) for c in kept_chunks)
                print(f"  {rows_seen:,} rows scanned, {found:,} kept")
    finally:
        close()

    if kept_chunks:
        out = pd.concat(kept_chunks, ignore_index=True)
    else:
        out = pd.DataFrame(columns=KEEP_COLUMNS + ["matched_species"])

    # One compound can be reported from several of these plants, which is a
    # real finding rather than a duplicate. The rows are collapsed on the
    # COCONUT identifier and the species are merged into one field, so each
    # compound is scored once but its plant list is preserved.
    stats = {"rows_scanned": rows_seen, "rows_matched": len(out)}
    if len(out):
        out = _collapse_duplicates(out)
        out = fill_missing_names(out)
    stats["unique_compounds"] = len(out)
    return out, stats


def fill_missing_names(df):
    # A third of COCONUT's natural products have no trivial name, only an
    # identifier. Left as NaN they would print as "nan" in the screening output
    # and in the demo app, which reads as a broken row rather than as a real
    # compound that simply has not been named.
    df = df.copy()
    missing = df["name"].isna() | (df["name"].astype(str).str.strip() == "")
    df.loc[missing, "name"] = df.loc[missing, "identifier"].astype(str)
    return df


def _collapse_duplicates(df):
    def merge_species(series):
        species = set()
        for value in series:
            species.update(str(value).split("|"))
        return "|".join(sorted(s for s in species if s))

    aggregations = {c: "first" for c in df.columns if c != "identifier"}
    aggregations["matched_species"] = merge_species
    collapsed = df.groupby("identifier", as_index=False).agg(aggregations)
    return collapsed.reset_index(drop=True)


def species_counts(df):
    # How many compounds came from each plant. Compounds shared between plants
    # are counted under each, so the total exceeds the number of rows.
    counts = {}
    for value in df["matched_species"]:
        for species in str(value).split("|"):
            if species:
                counts[species] = counts.get(species, 0) + 1
    return pd.Series(counts).sort_values(ascending=False)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Filter COCONUT to jamu species")
    parser.add_argument("zip_path", help="path to the COCONUT bulk csv zip")
    parser.add_argument("out_csv", help="where to write the filtered compounds")
    parser.add_argument("--chunksize", type=int, default=50000)
    args = parser.parse_args()

    print(f"scanning {args.zip_path}")
    compounds, stats = filter_bulk(args.zip_path, chunksize=args.chunksize)
    print(f"\nrows scanned:     {stats['rows_scanned']:,}")
    print(f"rows matched:     {stats['rows_matched']:,}")
    print(f"unique compounds: {stats['unique_compounds']:,}")

    if len(compounds):
        print("\nper species:")
        print(species_counts(compounds).to_string())

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    compounds.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}")


def lookup_by_name(bulk_path, wanted_names, chunksize=50000):
    # Pull specific compounds out of COCONUT by name.
    #
    # This exists so that reference compounds (positive controls, literature
    # examples) come from the database like everything else. Typing a SMILES
    # string in by hand produces a structure with no identifier and no
    # provenance, which cannot be cited and cannot be checked.
    #
    # Matching is exact on the lowercased name, not a substring test: searching
    # for "piperine" as a substring also returns piperidine alkaloids, which
    # are different compounds entirely.
    wanted = {str(n).strip().lower() for n in wanted_names}

    reader, close = _bulk_reader(
        bulk_path, None, chunksize, lambda c: c in KEEP_COLUMNS
    )

    hits = []
    try:
        for chunk in reader:
            if "name" not in chunk.columns:
                break
            lowered = chunk["name"].fillna("").str.strip().str.lower()
            found = chunk[lowered.isin(wanted)]
            if len(found):
                hits.append(found.copy())
    finally:
        close()

    if not hits:
        return pd.DataFrame(columns=KEEP_COLUMNS)

    out = pd.concat(hits, ignore_index=True)
    # One name can appear on several COCONUT entries (different stereochemistry
    # or source records). Keep the first per name and say how many were merged.
    out["_name_lower"] = out["name"].str.strip().str.lower()
    out = out.drop_duplicates("_name_lower").drop(columns="_name_lower")
    return out.reset_index(drop=True)
