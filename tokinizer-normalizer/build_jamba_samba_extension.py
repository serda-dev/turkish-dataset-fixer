#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path


TURKISH_CHARS = set("çğıİöşüÇĞÖŞÜ")
WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]+")
TOKEN_RE = re.compile(r"^▁?[A-Za-zÇĞİÖŞÜçğıöşü]+$")
SPECIAL_RE = re.compile(r"^<[^>]+>$")
CURATED_SUFFIXES = {
    "lar",
    "ler",
    "dır",
    "dir",
    "dur",
    "dür",
    "tır",
    "tir",
    "tur",
    "tür",
    "dan",
    "den",
    "tan",
    "ten",
    "lık",
    "lik",
    "luk",
    "lük",
    "yor",
    "acak",
    "ecek",
    "miş",
    "mış",
    "muş",
    "müş",
    "siniz",
    "sınız",
    "sunuz",
    "sünüz",
    "ımız",
    "imiz",
    "umuz",
    "ümüz",
    "ları",
    "leri",
    "larına",
    "lerine",
    "larından",
    "lerinden",
    "acağı",
    "eceği",
    "acaklar",
    "ecekler",
    "acaktır",
    "ecektir",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_samba_merges(raw_merges):
    merges = []
    for item in raw_merges:
        if isinstance(item, str):
            left, right = item.split(" ", 1)
            merges.append((left, right))
        else:
            merges.append(tuple(item))
    return merges


def extract_text(record):
    if isinstance(record, dict):
        for key in ("text", "content", "body"):
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(record, str):
        return record
    return None


def build_word_frequency(input_dir: Path, max_lines_per_file: int):
    counter = Counter()
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if idx >= max_lines_per_file:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = extract_text(record)
                if not text:
                    continue
                for word in WORD_RE.findall(text):
                    counter[word] += 1
                    counter[word.casefold()] += 1
    return counter


def is_turkish_relevant(token: str):
    plain = token[1:] if token.startswith("▁") else token
    if not plain or len(plain) < 3:
        return False
    if SPECIAL_RE.match(token):
        return False
    if not TOKEN_RE.match(token):
        return False
    return any(ch in plain for ch in TURKISH_CHARS) or plain.casefold() in CURATED_SUFFIXES


def build_parent_map(merges):
    parent_map = {}
    for index, (left, right) in enumerate(merges):
        parent_map[left + right] = (left, right, index)
    return parent_map


def compute_dependency_closure(token, known_vocab, known_merges, parent_map, memo):
    memo_key = (token, len(known_vocab), len(known_merges))
    if memo_key in memo:
        return memo[memo_key]

    planned_tokens = []
    planned_merges = []
    visiting = set()

    def walk(current):
        if current in known_vocab or current in planned_tokens:
            return True
        if current in visiting:
            return False
        parent = parent_map.get(current)
        if parent is None:
            return False
        visiting.add(current)
        left, right, merge_index = parent
        if not walk(left) or not walk(right):
            visiting.remove(current)
            return False
        visiting.remove(current)
        merge = (left, right, merge_index)
        if (left, right) not in known_merges and all((left, right) != (m[0], m[1]) for m in planned_merges):
            planned_merges.append(merge)
        if current not in known_vocab and current not in planned_tokens:
            planned_tokens.append(current)
        return True

    ok = walk(token)
    result = (ok, planned_tokens, planned_merges)
    memo[memo_key] = result
    return result


def build_candidates(samba_vocab, base_vocab, word_frequency):
    candidates = []
    for token, vocab_id in samba_vocab.items():
        if token in base_vocab:
            continue
        if not is_turkish_relevant(token):
            continue
        plain = token[1:] if token.startswith("▁") else token
        freq = word_frequency.get(plain, 0) + word_frequency.get(plain.casefold(), 0)
        has_turkish = any(ch in plain for ch in TURKISH_CHARS)
        suffix_bonus = 2000 if plain.casefold() in CURATED_SUFFIXES else 0
        prefix_bonus = 150 if token.startswith("▁") else 0
        rank_bonus = max(0, 5000 - vocab_id) / 10
        score = (freq * 100) + suffix_bonus + prefix_bonus + rank_bonus + (200 if has_turkish else 0)
        if score <= 0:
            continue
        candidates.append(
            {
                "token": token,
                "plain": plain,
                "samba_id": vocab_id,
                "frequency": freq,
                "score": score,
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["samba_id"], item["token"]))
    return candidates


def select_extensions(base_vocab, base_merges, samba_vocab, samba_merges, word_frequency, max_new_tokens):
    parent_map = build_parent_map(samba_merges)
    known_vocab = set(base_vocab)
    known_merges = {tuple(item) for item in base_merges}
    selected_tokens = []
    selected_merges = []
    selected_token_set = set()
    selected_merge_set = set()
    rejected = []
    memo = {}

    candidates = build_candidates(samba_vocab, base_vocab, word_frequency)
    for candidate in candidates:
        if len(selected_tokens) >= max_new_tokens:
            break
        ok, needed_tokens, needed_merges = compute_dependency_closure(
            candidate["token"],
            known_vocab,
            known_merges,
            parent_map,
            memo,
        )
        if not ok:
            rejected.append({"token": candidate["token"], "reason": "unresolvable_dependency"})
            continue

        new_unique_tokens = [tok for tok in needed_tokens if tok not in known_vocab and tok not in selected_token_set]
        if not new_unique_tokens:
            continue
        if len(selected_tokens) + len(new_unique_tokens) > max_new_tokens:
            continue

        for left, right, merge_index in sorted(needed_merges, key=lambda item: item[2]):
            merge_pair = (left, right)
            if merge_pair in known_merges or merge_pair in selected_merge_set:
                continue
            selected_merges.append(
                {
                    "left": left,
                    "right": right,
                    "samba_merge_index": merge_index,
                    "result": left + right,
                }
            )
            selected_merge_set.add(merge_pair)
            known_merges.add(merge_pair)

        for token in new_unique_tokens:
            if token in known_vocab or token in selected_token_set:
                continue
            selected_tokens.append(
                {
                    "token": token,
                    "source_samba_id": samba_vocab[token],
                    "plain": token[1:] if token.startswith("▁") else token,
                }
            )
            selected_token_set.add(token)
            known_vocab.add(token)

    return selected_tokens, selected_merges, rejected, candidates


def assign_new_ids(base_vocab, selected_tokens):
    next_id = max(base_vocab.values()) + 1
    new_vocab_entries = {}
    for item in selected_tokens:
        new_vocab_entries[item["token"]] = next_id
        item["new_id"] = next_id
        next_id += 1
    return new_vocab_entries


def write_outputs(
    output_dir: Path,
    base_tokenizer,
    base_config_path: Path,
    selected_tokens,
    selected_merges,
    candidates,
    rejected,
    sampled_lines,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded = json.loads(json.dumps(base_tokenizer))
    new_vocab_entries = assign_new_ids(expanded["model"]["vocab"], selected_tokens)
    expanded["model"]["vocab"].update(new_vocab_entries)
    expanded["model"]["merges"].extend([[item["left"], item["right"]] for item in selected_merges])

    with (output_dir / "tokenizer.json").open("w", encoding="utf-8") as handle:
        json.dump(expanded, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    shutil.copy2(base_config_path, output_dir / "tokenizer_config.json")

    report = {
        "base_vocab_size": len(base_tokenizer["model"]["vocab"]),
        "base_merges_size": len(base_tokenizer["model"]["merges"]),
        "new_vocab_size": len(expanded["model"]["vocab"]),
        "new_merges_size": len(expanded["model"]["merges"]),
        "added_token_count": len(selected_tokens),
        "added_merge_count": len(selected_merges),
        "sampled_lines_per_file": sampled_lines,
        "selected_tokens_preview": selected_tokens[:200],
        "selected_merges_preview": selected_merges[:200],
        "top_candidate_preview": candidates[:200],
        "rejected_preview": rejected[:200],
    }
    with (output_dir / "extension_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build a conservative Jamba base tokenizer extension from SambaLingo.")
    root = Path(__file__).resolve().parent
    parser.add_argument("--base-tokenizer", type=Path, default=root / "tokenizer_1_Base.json")
    parser.add_argument("--base-config", type=Path, default=root / "tokenizer1_base_config.json")
    parser.add_argument("--samba-tokenizer", type=Path, default=root / "SambaLingo" / "tokenizer.json")
    parser.add_argument("--input-dir", type=Path, default=root.parent / "input")
    parser.add_argument("--output-dir", type=Path, default=root / "jamba_samba_extended")
    parser.add_argument("--max-lines-per-file", type=int, default=20000)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()

    base_tokenizer = load_json(args.base_tokenizer)
    samba_tokenizer = load_json(args.samba_tokenizer)
    samba_merges = parse_samba_merges(samba_tokenizer["model"]["merges"])
    word_frequency = build_word_frequency(args.input_dir, args.max_lines_per_file)
    selected_tokens, selected_merges, rejected, candidates = select_extensions(
        base_vocab=base_tokenizer["model"]["vocab"],
        base_merges=base_tokenizer["model"]["merges"],
        samba_vocab=samba_tokenizer["model"]["vocab"],
        samba_merges=samba_merges,
        word_frequency=word_frequency,
        max_new_tokens=args.max_new_tokens,
    )
    write_outputs(
        output_dir=args.output_dir,
        base_tokenizer=base_tokenizer,
        base_config_path=args.base_config,
        selected_tokens=selected_tokens,
        selected_merges=selected_merges,
        candidates=candidates,
        rejected=rejected,
        sampled_lines=args.max_lines_per_file,
    )

    print("added_tokens", len(selected_tokens))
    print("added_merges", len(selected_merges))
    if selected_tokens:
        print("first_added_tokens", selected_tokens[:20])


if __name__ == "__main__":
    main()
