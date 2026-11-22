# Tidy up drug names so two lists can be matched against each other.
#
# The problem: the same drug gets written differently by different sources.
# "Amiodarone", "amiodarone HCl", "Amiodarone hydrochloride (oral)" are all one
# drug. Match on the raw text and they look like three different drugs, so rows
# quietly fail to join.
#
# Why that is dangerous here: a drug that fails to match just disappears from
# the substrate table, which the screen then reads as "this drug is not cleared
# by CYP3A4". A silent miss turns into a confident wrong answer.
#
# So: lowercase everything, throw away bracketed notes, and strip salt words
# from the END of the name. Deliberately nothing cleverer than that. No
# guessing at typos, because wrongly merging two different drugs is worse than
# reporting one that did not match.

import re

# Words that get stuck on the end of a drug name to say which salt form it is.
# They are stripped repeatedly until nothing more comes off, so
# "amiodarone hydrochloride monohydrate" ends up as just "amiodarone".
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

# Notes in brackets like "(oral)" or "[Tikosyn]" say nothing about which drug
# it is, and the two sources write them differently anyway.
_BRACKETS = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}")

# Turn anything that is not a letter, digit or space into a space, so that
# "co-trimoxazole" and "co trimoxazole" end up identical.
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalize_drug_name(name):
    # One drug name in, one clean matching key out. "" if nothing survives.
    if name is None:
        return ""
    text = str(name).lower()
    text = _BRACKETS.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    if not text:
        return ""

    # Only strip salt words from the END. A salt word at the START is part of
    # how the source actually names the drug, as in "sodium valproate".
    # Stripping it there would leave "valproate", which still does not match
    # "valproic acid", so nothing is gained and a real name could be destroyed.
    tokens = text.split(" ")
    while len(tokens) > 1 and tokens[-1] in _SALT_WORDS:
        tokens.pop()
    return " ".join(tokens)


def build_name_index(names):
    # Clean key -> the first original name that produced it.
    # Lets us report what a matched row was actually called in the source.
    index = {}
    for name in names:
        key = normalize_drug_name(name)
        if key and key not in index:
            index[key] = name
    return index


def match_report(left_names, right_names):
    # Compare two lists of names after cleaning, and report what matched and
    # what did not. The misses are returned, not silently dropped, so they can
    # be looked at by hand.
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
