#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safely send matched files to trash (instead of permanent deletion).

Example:
  # 先预览（默认 dry-run）
  python safe_clean_buffers.py --root ./rl/result/lunarlander

  # 确认无误后真正执行
  python safe_clean_buffers.py --root ./rl/result/lunarlander --apply

  # 递归更深
  python safe_clean_buffers.py --root ./rl/result/lunarlander --depth 4 --apply

  # 自定义正则
  python safe_clean_buffers.py --root ./rl/result/lunarlander --pattern '^.+_buffer\\d+\\.pkl$' --apply
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    from send2trash import send2trash
except ImportError:
    send2trash = None


DEFAULT_PATTERN = r"^.+_buffer\d+\.pkl$"


def iter_paths(root: Path, depth: int):
    """Yield paths under root up to a given depth (depth>=1)."""
    if depth <= 0:
        return
    # depth=1: root/* ; depth=2: root/**/* up to 2, etc.
    # 这里用递归生成器避免一次性 glob 巨量文件
    def _walk(p: Path, d: int):
        if d == 0:
            return
        try:
            for child in p.iterdir():
                yield child
                if child.is_dir():
                    yield from _walk(child, d - 1)
        except PermissionError:
            return

    yield from _walk(root, depth)


def main():
    parser = argparse.ArgumentParser(
        description="Safely clean buffer files by sending them to trash.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=str, required=True, help="Root directory to scan.")
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN, help="Regex to match file names.")
    parser.add_argument(
        "--depth", type=int, default=2,
        help="Max directory depth to scan under root (1 means only direct children)."
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually send files to trash. Without this flag, it is dry-run."
    )
    parser.add_argument(
        "--only-critic-actor-dirs", action="store_true",
        help="Only delete inside directories whose name contains both 'critic' and 'actor'."
    )
    parser.add_argument(
        "--path-guard", type=str, default="",
        help="Safety guard: require root resolved path to start with this prefix (e.g. /home/geert/rl/)."
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists() or not root.is_dir():
        print(f"[ERR] root not found or not a directory: {root}", file=sys.stderr)
        return 2

    root_resolved = root.resolve()
    if args.path_guard:
        guard = Path(args.path_guard).expanduser().resolve().as_posix()
        if not root_resolved.as_posix().startswith(guard):
            print(f"[ERR] path-guard failed.\n  root:  {root_resolved}\n  guard: {guard}", file=sys.stderr)
            return 2

    if not args.apply:
        mode = "DRY-RUN"
    else:
        mode = "APPLY"

    if args.apply and send2trash is None:
        print("[ERR] send2trash not installed. Run: pip install send2trash", file=sys.stderr)
        return 2

    try:
        name_re = re.compile(args.pattern)
    except re.error as e:
        print(f"[ERR] invalid regex pattern: {e}", file=sys.stderr)
        return 2

    # 可选：只在 critic+actor 目录内部删除
    def eligible_context(p: Path) -> bool:
        if not args.only_critic_actor_dirs:
            return True
        # 只要任意父目录（含自身父）名字同时含 critic 和 actor 就算 eligible
        # 这样你不用假设层级结构固定
        for parent in p.parents:
            if parent == root_resolved.parent:
                break
            n = parent.name.lower()
            if "critic" in n and "actor" in n:
                return True
        return False

    matched = []
    scanned = 0

    for p in iter_paths(root_resolved, args.depth):
        scanned += 1
        if not p.is_file():
            continue
        if not eligible_context(p):
            continue
        if name_re.match(p.name):
            matched.append(p)

    print(f"[{mode}] root={root_resolved}")
    print(f"[{mode}] pattern={args.pattern}  depth={args.depth}")
    if args.only_critic_actor_dirs:
        print(f"[{mode}] only inside dirs containing both 'critic' and 'actor'")
    if args.path_guard:
        print(f"[{mode}] path_guard={Path(args.path_guard).expanduser().resolve()}")

    print(f"[{mode}] scanned_entries={scanned}  matched_files={len(matched)}")

    if not matched:
        return 0

    # 先输出清单
    for p in matched:
        print(f"[{mode}] match: {p}")

    if not args.apply:
        print(f"[{mode}] No files were deleted. Re-run with --apply to send to trash.")
        return 0

    # 真执行
    ok = 0
    fail = 0
    for p in matched:
        try:
            send2trash(str(p))
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[ERR] failed to trash: {p}  ({e})", file=sys.stderr)

    print(f"[APPLY] trashed_ok={ok}  trashed_fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
