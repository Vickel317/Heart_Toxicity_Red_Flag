# Drug name normalisation, shared by the lookup tables and the screen.
#
# The FDA substrate table and the CredibleMeds QT list are maintained by
# different people and disagree about spelling, salt forms and capitalisation.
# Joining them on the raw name silently loses rows, and a lost row here looks
# exactly like "this drug is not a substrate", which is the wrong answer.
#
# Normalisation is deliberately conservative: it lowercases, drops bracketed
# notes and strips TRAILING salt and hydrate words. It does not try to guess
# at misspellings, because a wrong merge is worse than a reported miss.

import re

# Salt and hydrate words that appear after a drug name. Order does not matter,
# they are stripped repeatedly until nothing more comes off, so
# "amiodarone hydrochloride monohydrate" reduces to "amiodarone".
_SALT_WORDS = {
    "hydrochloride", "hcl", "dihydrochloride",
    "hydrobromide", "bromide", "chloride", "iodide",
    "mesylate", "mesilate", "besylate", "besilate", "tosylate", "napsylate",
    "maleate", "fumarate", "succinate", "tartrate", "bitartrate",
    "citrate", "dihydrogencitrate", "lactate", "gluconate", "glucuronate",
    "sulfate", "sulphate", "bisulfate", "hemisulfate",
    "phosphate", "diphosphate", "hydrogenphosphate",
    "acetate", "trifluoroacetate", "propionate", "butyrate", "valerate",
    "benzoate", "salicylate", "stearate", "palmitate", "laurate",
    "oxalate", "malate", "malonate", "carbonate", "bicarbonate",
    "nitrate", "nitrite", "pamoate", "embonate", "xinafoate",
    "furoate", "dipropionate", "propionate", "undecylenate",
    "sodium", "potassium", "calcium", "magnesium", "zinc", "aluminium",
    "monohydrate", "dihydrate", "trihydrate", "hemihydrate", "hydrate",
    "anhydrous", "base", "free",
}

# Bracketed notes such as "(oral)" or "[Tikosyn]" carry no identity information
# and differ between the two sources.
_BRACKETS = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}")

# Anything that is not a letter, digit or space becomes a space, so
# "co-trimoxazole" and "co trimoxazole" land on the same key.
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalize_drug_name(name):
    # Return the join key for one drug name, or "" if nothing is left.
    if name is None:
        return ""
    text = str(name).lower()
    text = _BRACKETS.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    if not text:
        return ""

    # Strip trailing salt words only. A leading salt word is part of the name
    # as the source writes it, as in "sodium valproate", and removing it there
    # would leave "valproate", which is a different key from "valproic acid"
    # anyway, so nothing is gained and a real name could be destroyed.
    tokens = text.split(" ")
    while len(tokens) > 1 and tokens[-1] in _SALT_WORDS:
        tokens.pop()
    return " ".join(tokens)


def build_name_index(names):
    # Map normalised key -> first original name that produced it.
    # Used to report what a matched row actually came from.
    index = {}
    for name in names:
        key = normalize_drug_name(name)
        if key and key not in index:
            index[key] = name
    return index


def match_report(left_names, right_names):
    # Compare two name lists after normalisation and return what joined and
    # what did not, so the misses can be fixed by hand instead of vanishing.
    left_index = build_name_index(left_names)
    right_index = build_name_index(right_names)
    left_keys = set(left_index)
    right_keys = set(right_index)
    shared = left_keys & right_keys
    return {
        "matched": sorted(shared),
        "left_only": sorted(left_keys - right_keys),
        "right_only": sorted(right_keys - left_keys),
        "left_index": left_index,
        "right_index": right_index,
    }
