# round_table_puzzle_script.py
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random, itertools, os

# Up to 20 classic placeholder names (extend if you want wider tables)
NAMES = [
    "Alice","Bob","Carol","Dave","Erin","Frank","Grace","Heidi","Ivan","Judy",
    "Mallory","Niaj","Olivia","Peggy","Quentin","Rupert","Sybil","Trent","Uma","Victor"
]

def random_bitstring(n: int, p: float = 0.5) -> List[int]:
    """Random length-n bitstring with Bernoulli(p); avoid all-zeros to keep the story non-trivial."""
    while True:
        bits = [1 if random.random() < p else 0 for _ in range(n)]
        if any(bits):
            return bits

def all_patterns(r: int):
    """All neighbourhood patterns of length 2r+1 in descending lexicographic order (e.g., 111.. -> 000..)."""
    L = 2 * r + 1
    return list(sorted(itertools.product([0, 1], repeat=L), reverse=True))

def sample_truth_table(r: int) -> Dict[Tuple[int, ...], int]:
    """Random Boolean rule for radius r; avoids degenerate all-0 or all-1."""
    table = {pat: random.randint(0, 1) for pat in all_patterns(r)}
    if len(set(table.values())) == 1:
        k = next(iter(table))
        table[k] = 1 - table[k]
    return table

def apply_rule(state: List[int], table: Dict[Tuple[int, ...], int], r: int) -> List[int]:
    """Synchronous update with circular wrap-around. Window order: [i-r, ..., i, ..., i+r]."""
    W = len(state)
    nxt = []
    for i in range(W):
        window = tuple(state[(i + j) % W] for j in range(-r, r + 1))
        nxt.append(table[window])
    return nxt

def bits_to_str(bits: List[int]) -> str:
    return "".join(map(str, bits))

def format_rule_table(table: Dict[Tuple[int, ...], int], r: int) -> str:
    header = f"NEIGHBOURHOOD ORDER: [left {r} .. left 1, self, right 1 .. right {r}] (circular)\n"
    lines = [header, "TRUTH TABLE (pattern -> next bit):"]
    for pat in all_patterns(r):
        lines.append(f"{''.join(map(str, pat))} -> {table[pat]}")
    return "\n".join(lines)

def seating_names(W: int) -> List[str]:
    """First W names; if ever W>len(NAMES) add suffixes or extend NAMES."""
    return NAMES[:W] if W <= len(NAMES) else [f"Player{i+1}" for i in range(W)]

def story_round_line(round_idx: int, bits: List[int], names: List[str]) -> str:
    raisers = [names[i] for i, b in enumerate(bits) if b == 1]
    if not raisers:
        return f"- Round {round_idx}. No one raises a hand; everyone keeps their hands on the table."
    if len(raisers) == 1:
        return f"- Round {round_idx}. Only {raisers[0]} raises a hand. Everyone else keeps their hands on the table."
    if len(raisers) == len(names):
        return f"- Round {round_idx}. Everyone raises a hand."
    who = " and ".join(raisers) if len(raisers) == 2 else (", ".join(raisers[:-1]) + f", and {raisers[-1]}")
    return f"- Round {round_idx}. {who} raise their hands. The others keep their hands on the table."

def build_prompt(names: List[str], rounds_bits: List[List[int]], shift: int = 1) -> str:
    W, T = len(names), len(rounds_bits)
    intro = (
        "You peek through a doorway into a cosy room.\n"
        f"{W} friends sit around a round table in this order: "
        + ", ".join(names[:-1]) + f", and {names[-1]} — and then back to {names[0]} again.\n"
        "They don’t talk. At the end of each round they all decide, at the very same moment, "
        "either to raise a hand or to keep both hands on the table.\n\n"
        "You watch and jot down what happens:\n"
    )
    lines = [intro] + [story_round_line(t, bits, names) for t, bits in enumerate(rounds_bits, start=1)]
    lines.append(
        "\nNow it’s your turn to be the clever observer.\n"
        f"Puzzle: What will each friend do in Round {T+shift}?\n"
        "Please answer in plain words, going in order around the table, starting from the first name above."
    )
    return "".join(lines)

@dataclass
class GameSpec:
    width: int
    radius: int
    rounds: int
    seed: Optional[int] = None
    shift: int = 1

def generate_game(spec: GameSpec):
    if spec.seed is not None:
        random.seed(spec.seed)
    W, r, T, shift = spec.width, spec.radius, spec.rounds, spec.shift
    names = seating_names(W)
    table = sample_truth_table(r)
    s0 = random_bitstring(W, p=0.5)
    orbit = [s0]
    for _ in range(T - 1):
        orbit.append(apply_rule(orbit[-1], table, r))
    next_state = orbit[-1]
    for _ in range(shift):
        next_state = apply_rule(next_state, table, r)
    return {
        "names": names,
        "width": W,
        "radius": r,
        "rounds": T,
        "shift": shift,
        "rule_table": format_rule_table(table, r),
        "orbit_strings": [bits_to_str(bits) for bits in orbit],
        "prompt": build_prompt(names, orbit, shift=spec.shift),
        "answer_next_bits": bits_to_str(next_state),
        "answer_next_names": [names[i] for i, b in enumerate(next_state) if b == 1],
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a round table puzzle.")
    parser.add_argument("--width", type=int, default=5, help="Number of friends around the table.")
    parser.add_argument("--radius", type=int, default=1, help="Radius of the cellular automaton rule (1 or 2).")
    parser.add_argument("--rounds", type=int, default=10, help="Number of observed rounds.")
    parser.add_argument("--shift", type=int, default=1, help="Number of steps to predict ahead.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    game = generate_game(GameSpec(width=args.width, radius=args.radius, rounds=args.rounds, seed=args.seed, shift=args.shift))

    print("--- PROMPT ---")
    print(game['prompt'])
    print("\n--- ANSWER (DO NOT PASTE IN LLM) ---")
    print("Next state bitstring:", game['answer_next_bits'])
    print("Next state hands up:", ", ".join(game['answer_next_names']) if game['answer_next_names'] else "No one")