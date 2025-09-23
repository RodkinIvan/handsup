# Wolfram class assignment for elementary cellular automata (ECA)
# Expanded to all 256 rules using reflection + complementation equivalence.

# Representative mapping (from Wolfram’s classification)
rep_to_class = {
    # Class 1
    0: 1, 8: 1, 32: 1, 40: 1, 128: 1, 136: 1, 160: 1, 168: 1,

    # Class 2
    1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 9: 2, 10: 2, 11: 2,
    12: 2, 13: 2, 14: 2, 15: 2, 19: 2, 23: 2, 24: 2, 25: 2, 26: 2,
    27: 2, 28: 2, 29: 2, 33: 2, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2,
    42: 2, 43: 2, 44: 2, 46: 2, 50: 2, 51: 2, 56: 2, 57: 2, 58: 2,
    62: 2, 72: 2, 73: 2, 74: 2, 76: 2, 77: 2, 78: 2, 94: 2, 104: 2,
    108: 2, 130: 2, 132: 2, 134: 2, 138: 2, 140: 2, 142: 2, 152: 2,
    154: 2, 156: 2, 162: 2, 164: 2, 170: 2, 172: 2, 178: 2, 184: 2,
    200: 2, 204: 2, 232: 2,

    # Class 3
    18: 3, 22: 3, 30: 3, 45: 3, 60: 3, 90: 3, 105: 3, 122: 3,
    126: 3, 146: 3, 150: 3,

    # Class 4
    41: 4, 54: 4, 106: 4, 110: 4,
}

# --- Helper functions ---
def rule_to_bits(rule):
    """Return list of 8 bits for rule (b7..b0)."""
    return [(rule >> i) & 1 for i in range(7, -1, -1)]

def bits_to_rule(bits):
    """Convert list of 8 bits [b7..b0] back to rule number."""
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val

def reflect_bits(bits):
    """Reflect neighborhoods: abc -> cba."""
    new_bits = [0] * 8
    for idx in range(8):
        nb = 7 - idx  # neighborhood numeric value
        a, b, c = (nb >> 2) & 1, (nb >> 1) & 1, nb & 1
        refl_nb = (c << 2) | (b << 1) | a
        dest_pos = 7 - refl_nb
        new_bits[dest_pos] = bits[idx]
    return new_bits

def complement_bits(bits):
    return [1 - b for b in bits]

def equivalents_of_rule(rule):
    """Return the set of all equivalent rules under reflection/complement."""
    b = rule_to_bits(rule)
    eq = set()
    eq.add(bits_to_rule(b))
    eq.add(bits_to_rule(reflect_bits(b)))
    bc = complement_bits(b)
    eq.add(bits_to_rule(bc))
    eq.add(bits_to_rule(reflect_bits(bc)))
    return eq

# --- Build full 256 mapping ---
full_map = {}
rep_equivs = {rep: equivalents_of_rule(rep) for rep in rep_to_class}

for rep, cls in rep_to_class.items():
    for r in rep_equivs[rep]:
        full_map[r] = cls

# Fill missing ones by matching to equivalent sets
for r in range(256):
    if r not in full_map:
        eqset = equivalents_of_rule(r)
        for rep, cls in rep_to_class.items():
            if eqset & rep_equivs[rep]:
                full_map[r] = cls
                break


# --- Done ---
print("Number of rules classified:", len(full_map))
# Example: print first 20 mappings
for k in range(20):
    print(k, "->", full_map[k])

print(len(rep_to_class))
print(len(full_map))

if len(full_map) != 256:
    for k in range(256):
        if k not in full_map:
            print(k, "missing")
            print(equivalents_of_rule(k))
            print(rule_to_bits(k))

# The dict is now in `full_map`