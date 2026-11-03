"""PR-AUC tables of the ReProbe paper (arXiv 2511.06209 v5), transcribed as printed (Tier B).

Source: the arXiv HTML of v5 (https://arxiv.org/html/2511.06209v5), parsed table by table.
Table 1 = Qwen3-8B, Table 4 = Qwen3-1.7B (native thinking), Table 5 = Qwen3-32B (native
thinking), Table 9 = Phi-4.

Each table: `printed_columns` is the set order of the printed header; every row holds the eight
per-set values in that order, then ID, OOD and overall (as printed). Use `per_set()` to get
{set: value} keyed by our set names.

Table 5 note: comparing with the scores stored in ReProbe's released Qwen3-32B managers shows
that the printed Meetings/Calendar and StrategyQA/ScienceQA columns are swapped (e.g. the
stored ReProbe column gives meetings .698, calendar .734, strategyqa .303, scienceqa .302, the
printed row reads .734, .698, .302, .303 under Meetings, Calendar, StrQA, SciQA). `per_set()`
undoes the swap by default (`corrected=True`); the ID, OOD and overall means are unaffected.
Its GSM8K column also disagrees with the released stored scores for all five PRM rows (e.g.
Qwen2.5-Math-PRM-7B prints .930, the stored scores give .552); that is not corrected here, it is
reported by the consistency check (KNOWN_ISSUES).

Table 4 note: the ReProbe row's Meetings cell prints .505, but the released 7-epoch manager gives
.553 and the printed OOD (.474) and overall (.447) means are those of .553 (with .505 they would
be .464 and .441): the cell is a typo, corrected by `per_set()` (`cell_fix`). Table 4 has no UQ,
Skywork or H4 rows, and its PRM scores are not in the released managers.

Table 9 note: the Phi-4 planning columns (Trips, Meetings, Calendar) come from traces that are
not released; the five released sets (MATH, GSM8K, ProofNet, StrategyQA, ScienceQA) form the
"t9" group.
"""

from __future__ import annotations

TEST = ("math", "gsm8k", "proofnet", "trips", "meetings", "calendar", "strategyqa", "scienceqa")
T9 = ("math", "gsm8k", "proofnet", "strategyqa", "scienceqa")

# canonical method keys used across the three tables
RANDOM, MAXPROB, MAXENT, PPL = "Random", "MaxProb", "MaxEntropy", "Perplexity"
SELFCERT, CCP, PTRUE = "Self-Certainty", "CCP", "P(True)"
SEMENT, LEXSIM, DEGMAT = "Semantic Entropy", "Lexical Similarity", "Degree Matrix"
SKYWORK, H4 = "Skywork-PRM-1.5B", "H4-Qwen2.5-PRM-1.5B-0.2"
SHEPHERD, RLHF_DS, RLHF_MI = "Math-Shepherd-PRM-7B", "RLHFlow-PRM-Deepseek-8B", "RLHFlow-PRM-Mistral-8B"
UNIVERSAL, PRM800K, PRM7B = "Universal-PRM-Qwen2.5-Math-7B", "Qwen2.5-Math-7B-PRM800k", "Qwen2.5-Math-PRM-7B"
RP_AL_SELF, RP_AL_DS = "ReProbe Attn+Logit, self-anno", "ReProbe Attn+Logit, DeepSeek-anno"
RP_HS_SELF, RP_HS_DS = "ReProbe hidden states, self-anno", "ReProbe hidden states, DeepSeek-anno"
RP_HS_OSS, RP_AL_Q3 = "ReProbe hidden states, GPT-OSS-anno", "ReProbe Attn+Logit, Qwen3-8B-anno"

GROUP = {  # section of the printed table each method sits in
    RANDOM: "UQ", MAXPROB: "UQ", MAXENT: "UQ", PPL: "UQ", SELFCERT: "UQ", CCP: "UQ", PTRUE: "UQ",
    SEMENT: "UQ", LEXSIM: "UQ", DEGMAT: "UQ", SKYWORK: "PRM", H4: "PRM", SHEPHERD: "PRM",
    RLHF_DS: "PRM", RLHF_MI: "PRM", UNIVERSAL: "PRM", PRM800K: "PRM", PRM7B: "PRM",
    RP_AL_SELF: "ReProbe", RP_AL_DS: "ReProbe", RP_HS_SELF: "ReProbe", RP_HS_DS: "ReProbe",
    RP_HS_OSS: "ReProbe", RP_AL_Q3: "ReProbe",
}

TABLES: dict[str, dict] = {
    "qwen3_8b": {
        "table": "Table 1",
        "printed_columns": TEST,
        "column_fix": {},
        "rows": {
            # MATH GSM8k ProofNet Trips Meetings Calendar StrQA SciQA | ID OOD Overall
            RANDOM: (.173, .061, .153, .524, .588, .486, .116, .125, .129, .368, .278),
            MAXPROB: (.221, .106, .194, .578, .655, .483, .114, .259, .174, .418, .326),
            MAXENT: (.212, .122, .185, .545, .618, .443, .119, .227, .173, .390, .309),
            PPL: (.205, .099, .176, .519, .572, .418, .110, .228, .160, .369, .291),
            SELFCERT: (.213, .101, .155, .516, .643, .482, .120, .243, .156, .401, .309),
            CCP: (.250, .090, .168, .584, .645, .452, .119, .235, .169, .407, .318),
            PTRUE: (.164, .059, .172, .535, .608, .490, .126, .263, .132, .404, .302),
            SEMENT: (.257, .116, .173, .565, .610, .492, .111, .265, .182, .409, .324),
            LEXSIM: (.250, .119, .170, .569, .603, .490, .120, .254, .180, .407, .322),
            DEGMAT: (.227, .089, .147, .534, .597, .484, .107, .258, .154, .396, .305),
            SKYWORK: (.283, .412, .147, .433, .532, .502, .254, .408, .281, .426, .371),
            H4: (.259, .171, .159, .597, .633, .495, .213, .228, .196, .433, .344),
            SHEPHERD: (.380, .405, .147, .662, .660, .657, .284, .415, .311, .536, .451),
            RLHF_DS: (.289, .540, .136, .583, .579, .504, .390, .518, .322, .515, .442),
            RLHF_MI: (.233, .537, .118, .523, .555, .499, .349, .415, .296, .468, .404),
            UNIVERSAL: (.534, .624, .329, .730, .753, .691, .328, .330, .496, .566, .540),
            PRM800K: (.586, .613, .301, .708, .768, .727, .362, .404, .500, .594, .559),
            PRM7B: (.531, .702, .310, .711, .757, .745, .334, .429, .514, .595, .565),
            RP_AL_SELF: (.529, .594, .260, .735, .779, .779, .394, .404, .461, .618, .559),
            RP_AL_DS: (.465, .616, .243, .740, .802, .786, .395, .361, .441, .617, .551),
            RP_HS_SELF: (.558, .673, .264, .799, .819, .785, .381, .553, .498, .667, .604),
            RP_HS_DS: (.534, .650, .281, .793, .817, .789, .344, .450, .488, .639, .582),
        },
    },
    "qwen3_1_7b": {
        "table": "Table 4",
        "printed_columns": TEST,
        "column_fix": {},
        "cell_fix": {(RP_HS_OSS, "meetings"): .553},
        "rows": {
            SHEPHERD: (.049, .248, .234, .219, .481, .338, .265, .313, .177, .323, .268),
            RLHF_DS: (.063, .267, .228, .169, .498, .322, .449, .368, .186, .361, .296),
            RLHF_MI: (.047, .184, .164, .141, .457, .275, .273, .279, .132, .285, .228),
            UNIVERSAL: (.175, .331, .299, .272, .708, .446, .335, .329, .268, .418, .362),
            PRM800K: (.217, .392, .505, .270, .716, .587, .391, .336, .371, .460, .427),
            PRM7B: (.293, .270, .465, .328, .732, .605, .378, .293, .343, .467, .421),
            RP_HS_OSS: (.312, .433, .465, .334, .505, .604, .426, .451, .403, .474, .447),
        },
    },
    "qwen3_32b": {
        "table": "Table 5",
        "printed_columns": TEST,  # as printed; the data says two pairs are swapped (column_fix)
        "column_fix": {"meetings": "calendar", "calendar": "meetings",
                       "strategyqa": "scienceqa", "scienceqa": "strategyqa"},
        "rows": {
            SKYWORK: (.303, .886, .289, .420, .565, .508, .527, .265, .493, .457, .470),
            H4: (.133, .464, .149, .470, .325, .386, .129, .221, .249, .306, .285),
            UNIVERSAL: (.578, .903, .568, .737, .634, .637, .189, .303, .683, .500, .569),
            PRM800K: (.409, .894, .454, .573, .591, .594, .282, .350, .586, .478, .518),
            PRM7B: (.668, .930, .646, .696, .660, .676, .313, .315, .748, .532, .613),
            RP_HS_OSS: (.676, .535, .682, .752, .734, .698, .302, .303, .631, .558, .585),
        },
    },
    "phi4": {
        "table": "Table 9",
        "printed_columns": TEST,
        "column_fix": {},
        "rows": {
            RANDOM: (.106, .038, .082, .552, .463, .324, .172, .086, .075, .319, .228),
            MAXPROB: (.127, .084, .123, .618, .548, .380, .252, .158, .111, .391, .286),
            MAXENT: (.112, .079, .107, .585, .533, .362, .248, .135, .099, .373, .270),
            PPL: (.117, .066, .099, .557, .508, .323, .228, .143, .094, .352, .255),
            SKYWORK: (.219, .181, .185, .408, .467, .327, .237, .415, .195, .371, .305),
            H4: (.174, .061, .105, .534, .476, .434, .212, .116, .113, .354, .264),
            SHEPHERD: (.248, .188, .188, .747, .584, .489, .249, .327, .208, .479, .378),
            RLHF_DS: (.200, .263, .109, .558, .455, .343, .315, .440, .191, .422, .335),
            RLHF_MI: (.141, .195, .093, .462, .424, .309, .261, .311, .143, .353, .274),
            UNIVERSAL: (.485, .213, .263, .741, .559, .497, .321, .252, .320, .474, .416),
            PRM800K: (.474, .406, .238, .825, .599, .568, .355, .329, .373, .535, .474),
            PRM7B: (.427, .377, .240, .791, .594, .555, .333, .310, .348, .517, .453),
            RP_AL_Q3: (.404, .340, .155, .756, .646, .592, .347, .347, .300, .538, .448),
        },
    },
}

# paper method -> estimator column stored in ReProbe's released test managers (scores[...] of
# data/test/<set>.traces.jsonl), per backbone; used by the consistency check
STORED = {
    "qwen3_8b": {RANDOM: "RandomBaselineClaim", MAXPROB: "MaximumClaimProbability",
                 MAXENT: "MaxTokenEntropyClaim", PPL: "PerplexityClaim",
                 RP_AL_DS: "LuqClaimEstimatorDummy_claim",
                 H4: "PRM_HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2",
                 UNIVERSAL: "PRM_universalprm/Universal-PRM"},
    "qwen3_1_7b": {RANDOM: "RandomBaselineClaim", MAXPROB: "MaximumClaimProbability",
                   MAXENT: "MaxTokenEntropyClaim", PPL: "PerplexityClaim",
                   RP_HS_OSS: "LuqClaimEstimatorDummy_claim"},
    "qwen3_32b": {SKYWORK: "PRM_Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
                  H4: "PRM_HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2",
                  UNIVERSAL: "PRM_universalprm/Universal-PRM",
                  PRM800K: "PRM_Qwen/Qwen2.5-Math-7B-PRM800K", PRM7B: "PRM_Qwen/Qwen2.5-Math-PRM-7B",
                  RP_HS_OSS: "LuqClaimEstimatorDummy_claim"},
    "phi4": {RANDOM: "RandomBaselineClaim", MAXPROB: "MaximumClaimProbability",
             MAXENT: "MaxTokenEntropyClaim", PPL: "PerplexityClaim",
             SKYWORK: "PRM_Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
             H4: "PRM_HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2",
             SHEPHERD: "PRM_peiyi9979/math-shepherd-mistral-7b-prm",
             RLHF_DS: "PRM_RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
             RLHF_MI: "PRM_RLHFlow/Llama3.1-8B-PRM-Mistral-Data",
             UNIVERSAL: "PRM_universalprm/Universal-PRM",
             PRM800K: "PRM_Qwen/Qwen2.5-Math-7B-PRM800K", PRM7B: "PRM_Qwen/Qwen2.5-Math-PRM-7B",
             RP_AL_Q3: "LuqClaimEstimatorDummy_claim"},
}

# disagreements that are known and explained (backbone, method, set) -> reason
KNOWN_ISSUES = {
    # every PRM row of Table 5's GSM8K column disagrees with the released stored scores (e.g. PRM-7B
    # prints .930, the stored column gives .552) while the ReProbe row (.535) reproduces
    **{("qwen3_32b", m, "gsm8k"): "Table 5 GSM8K column of the PRM rows does not match the released stored scores"
       for m in (SKYWORK, H4, UNIVERSAL, PRM800K, PRM7B)},
    **{("phi4", m, s): "Table 9 planning columns come from traces that are not released"
       for m in TABLES["phi4"]["rows"] for s in ("trips", "meetings", "calendar")},
    # Table 1's UQ rows on four sets were computed on another label set than the released one:
    # the printed Random row (.173/.061/.153/.116 on MATH/GSM8K/ProofNet/StrategyQA) is far from
    # the released positive rates (.161/.086/.103/.189), while the ReProbe rows reproduce exactly
    **{("qwen3_8b", m, s): "Table 1 UQ rows on this set do not use the released labels (printed Random AP "
       "differs from the released positive rate)"
       for m in (RANDOM, MAXPROB, MAXENT, PPL) for s in ("math", "gsm8k", "proofnet", "strategyqa")},
}


def per_set(backbone: str, method: str, corrected: bool = True) -> dict[str, float]:
    """{set: printed PR-AUC} plus 'id', 'ood', 'overall' (and 't9' for Phi-4, the mean of the
    five released sets, computed from the printed cells). Empty dict if not in the table."""
    t = TABLES.get(backbone)
    if not t or method not in t["rows"]:
        return {}
    v = t["rows"][method]
    out = {}
    for col, x in zip(t["printed_columns"], v[:8]):
        out[t["column_fix"].get(col, col) if corrected else col] = float(x)
    if corrected:
        out.update({c: float(x) for (m, c), x in t.get("cell_fix", {}).items() if m == method})
    out.update({"id": float(v[8]), "ood": float(v[9]), "overall": float(v[10])})
    if backbone == "phi4":
        out["t9"] = float(sum(out[s] for s in T9) / len(T9))
    return out


def methods(backbone: str) -> list[str]:
    return list(TABLES.get(backbone, {}).get("rows", {}))
