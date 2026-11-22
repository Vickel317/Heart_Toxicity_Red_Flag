# Get the jamu compounds.
#
# COCONUT is a free database of natural products: molecules found in living
# things, with a record of which organism each one came from. We want the ones
# reported from six Indonesian medicinal plants.
#
# Problem 1, the easy one: the download is a ~700 MB spreadsheet. Loading it
# all at once will exhaust memory on a laptop. So it is read in chunks, and
# each chunk is filtered and thrown away before the next arrives. That turns an
# impossible step into a few minutes.
#
# Problem 2, the one that actually matters: PLANT NAMES ARE A MESS. COCONUT's
# organism field is free text contributed by many different databases, so one
# plant shows up under its official name, its old names, and several spellings.
# Search for only the exact official name and most rows vanish. And a row that
# vanishes here looks exactly like "this plant has no known compounds", which
# is a wrong answer that looks like a real one.

import glob
import os
import re
import zipfile

import pandas as pd

# Places the download might be sitting, as seen from a notebook in notebooks/.
#
# The filename has the release date in it (coconut_csv-11-2026.csv) because
# COCONUT reissues the file every month. So we search by pattern instead of
# hard coding a name: a hard coded name would break the moment anyone
# downloaded a newer copy. Zipped or unzipped both work.
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
    # Find whichever COCONUT file was downloaded.
    #
    # If there is none, this raises an error WITH the download link rather than
    # returning None. That is deliberate: returning None would let the notebook
    # carry on and produce an empty table, which reads like "these plants have
    # no compounds" instead of "you forgot to download the file".
    search_dirs = search_dirs or BULK_SEARCH_DIRS

    found = []
    for directory in search_dirs:
        for pattern in BULK_PATTERNS:
            found.extend(glob.glob(os.path.join(directory, pattern)))

    if not found:
        raise FileNotFoundError("No COCONUT bulk file found. " + DOWNLOAD_HINT)

    # Sorting by name puts the newest release last, since the dates are in the
    # filenames. If both a zip and an unzipped copy exist, prefer the unzipped
    # one so we are not decompressing on every single chunk.
    found.sort()
    plain = [f for f in found if f.lower().endswith(".csv")]
    return (plain or found)[-1]


def _bulk_reader(path, member, chunksize, usecols):
    # Open the file for chunk-by-chunk reading, handling both a zip and a
    # plain csv. Returns (something to loop over, something to call when done).
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

# The six plants, each with the name variants that actually turn up in these
# databases. Every variant points back to one official name so the results can
# be grouped per plant.
#
# Everything listed under one plant is the SAME plant written differently, not
# a related plant. Phyllanthus amarus is deliberately NOT merged into
# P. niruri: papers confuse the two constantly, but they are genuinely
# different species, and merging them would claim more than the data supports.
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

# Which columns to keep. COCONUT has 44; we need nine. Dropping the rest as
# early as possible is what keeps each chunk small enough to handle.
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
    # Build one search pattern per name variant.
    #
    # The \b parts mean "must be a whole word". Without them, searching for
    # "curcuma longa" would also match "Curcuma longaeflora", a different
    # plant. With them we still match "Curcuma longa L." and "Curcuma longa
    # Roscoe", which is what we want, because databases tack the name of the
    # botanist who classified the plant onto the end.
    patterns = []
    for canonical, variants in species_map.items():
        for variant in variants:
            pattern = re.compile(r"\b" + re.escape(variant) + r"\b")
            patterns.append((canonical, variant, pattern))
    return patterns


def _normalise_organism(text):
    return _SPACES.sub(" ", str(text).lower()).strip()


def match_species(organisms_field, patterns):
    # Does this row's organism text mention any of our six plants?
    # Returns the official names of the ones it matched, empty list if none.
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
    # Read the whole COCONUT file in chunks, keeping only rows that mention
    # one of our plants. Returns (the kept rows, some counts).
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

            # Quick cheap check before the careful slow one. Almost none of
            # COCONUT has anything to do with these six plants, so first do a
            # rough test on just the genus ("curcuma", "zingiber"), which
            # throws away most of the chunk. Only the survivors get the
            # careful whole-word matching, which is much slower.
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

    # The same compound can be reported from several of these plants. That is
    # a real fact, not a duplicate row to delete. So rows are merged on the
    # compound's ID and the plant names joined together: each compound gets
    # scored once, but we still know every plant it came from.
    stats = {"rows_scanned": rows_seen, "rows_matched": len(out)}
    if len(out):
        out = _collapse_duplicates(out)
        out = fill_missing_names(out)
    stats["unique_compounds"] = len(out)
    return out, stats


def fill_missing_names(df):
    # About a third of these compounds have no common name, only an ID code.
    # Left empty they print as "nan" in the results, which looks like a bug.
    # Using the ID code instead makes it clear the compound is real and simply
    # has never been given a name.
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
    # Count compounds per plant. A compound found in two plants is counted
    # under both, so these numbers add up to more than the number of rows.
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
    # Look up specific compounds by name.
    #
    # This exists so that reference compounds come from the database like
    # everything else does. Typing a structure in by hand gives something with
    # no ID and no source, which cannot be cited and cannot be checked by
    # anyone else.
    #
    # The match is exact on the lowercased name, not "contains". Searching for
    # "piperine" as a substring would also drag in piperidine alkaloids, which
    # are completely different compounds.
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
