"""
Ledger 层：快照 + 流水，按剧本项目分文件夹存储。
只负责持久化和检索，不做任何判断——判断权全在 brain。

目录结构（系列嵌套）：
- 独立项目（从没被续写过）：ledger/<name>/
- 系列（有续集）：ledger/<系列根，即第一幕的名字>/<每一幕的名字>/
  独立项目在第一次被续写时，会自动"晋升"成系列——把自己搬进以自己名字
  命名的系列文件夹里，新的一幕作为该系列文件夹下的另一个子文件夹。
  这样文件夹本身就能看出"哪些幕属于同一个系列"，不需要打开元数据才知道。

对外的项目标识符始终是"某一幕自己的名字"这个单一字符串（不是"系列/幕"这种
复合路径），调用方（brain / chat.py / server.py / 前端）不需要关心一个项目
究竟存在扁平位置还是嵌套位置，这层解析全部封装在本文件内部。
"""

import json
import os
import shutil
from datetime import datetime, timezone

LEDGER_ROOT = "ledger"


def _resolve_existing_path(project_name: str):
    """找 project_name 实际存在的磁盘路径。扁平和嵌套两种位置都会找，
    都找不到就返回 None（不创建任何东西——纯查询）。"""
    flat_path = os.path.join(LEDGER_ROOT, project_name)
    if os.path.isfile(os.path.join(flat_path, "snapshot.json")):
        return flat_path

    if os.path.isdir(LEDGER_ROOT):
        for entry in os.listdir(LEDGER_ROOT):
            nested = os.path.join(LEDGER_ROOT, entry, project_name)
            if os.path.isfile(os.path.join(nested, "snapshot.json")):
                return nested
    return None


def project_dir(project_name: str) -> str:
    """返回 project_name 的磁盘路径，如果它还不存在则按扁平方式新建
    （新项目在被续写之前，不知道它会不会成为系列的一部分，默认先扁平存）。"""
    existing = _resolve_existing_path(project_name)
    if existing:
        return existing
    flat_path = os.path.join(LEDGER_ROOT, project_name)
    os.makedirs(flat_path, exist_ok=True)
    return flat_path


def empty_snapshot() -> dict:
    return {
        "rule_based": {
            "genre": None,
            "tone": None,
            "length": None,
            "characters": [],
            "setting": {"location": None, "time_period": None},
            "must_include": [],
            "must_avoid": [],
        },
        "throughline": None,
        "candidates": [],
        "sub_intents": [],
        "assumptions": [],
        "raw_user_request": "",
        "_meta": {
            "project_state": "empty",
            "missing_critical_info": [],
            "flagged_injection": False,
            "flagged_snippets": [],
            "conflicts_flagged": [],
            "predecessor": None,
            "continuity_recap": None,
            "generated_from": {},  # {mode: 生成这一幕正文时，上一幕当时的最新草稿文件名}
        },
    }


def load_snapshot(project_name: str) -> dict:
    path = os.path.join(project_dir(project_name), "snapshot.json")
    if not os.path.exists(path):
        return empty_snapshot()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(project_name: str, snapshot: dict) -> None:
    path = os.path.join(project_dir(project_name), "snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def append_log(project_name: str, role: str, content) -> None:
    path = os.path.join(project_dir(project_name), "log.jsonl")
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "content": content,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def list_projects() -> list:
    """返回所有"幕"的名字（扁平的独立项目 + 每个系列文件夹里的每一幕），
    对外依然是一份单层的名字列表，嵌套只是磁盘层面的事。"""
    if not os.path.isdir(LEDGER_ROOT):
        return []
    names = []
    for entry in sorted(os.listdir(LEDGER_ROOT)):
        entry_path = os.path.join(LEDGER_ROOT, entry)
        if not os.path.isdir(entry_path):
            continue
        if os.path.isfile(os.path.join(entry_path, "snapshot.json")):
            names.append(entry)  # 扁平的独立项目
        else:
            for act in sorted(os.listdir(entry_path)):  # 系列文件夹，展开每一幕
                act_path = os.path.join(entry_path, act)
                if os.path.isfile(os.path.join(act_path, "snapshot.json")):
                    names.append(act)
    return names


def series_root(project_name: str) -> str:
    """顺着 predecessor 一路往回找到这条链最初的那一幕，即它所在系列文件夹的名字。
    没有 predecessor 的项目，它自己就是系列根（不管现在是扁平还是嵌套存储）。"""
    current = project_name
    seen = set()
    while current not in seen:
        seen.add(current)
        pred = load_snapshot(current)["_meta"].get("predecessor")
        if not pred:
            return current
        current = pred
    return current  # 理论上不会出现循环引用，这里只是防御一下


def project_summaries() -> list:
    """项目列表 + 状态 + 承接自哪个项目 + 所属系列根，供前端/CLI 展示用。"""
    summaries = []
    for name in list_projects():
        snap = load_snapshot(name)
        summaries.append({
            "name": name,
            "state": snap["_meta"]["project_state"],
            "predecessor": snap["_meta"].get("predecessor"),
            "series": series_root(name),
        })
    return summaries


def _migrate_chain_to_series(any_member_name: str) -> str:
    """确保 any_member_name 所在的整条 predecessor 链，都归到同一个系列文件夹下
    （系列文件夹名 = 这条链最初那一幕的名字）。已经在系列里的链调用这个是空操作。
    用于：①每次创建新续集前，保证整条链已经是嵌套结构；②修复历史遗留的扁平数据。
    """
    root = series_root(any_member_name)
    series_path = os.path.join(LEDGER_ROOT, root)

    root_current_path = _resolve_existing_path(root)
    root_already_nested = (
        root_current_path is not None
        and os.path.dirname(root_current_path) != LEDGER_ROOT
    )

    if not root_already_nested:
        # 把根节点从 ledger/<root>/ 搬成 ledger/<root>/<root>/，
        # 用一个临时名字中转，避免"把文件夹搬进自己底下"的路径冲突。
        if root_current_path is None:
            raise FileNotFoundError(f"project '{root}' not found")
        tmp_path = os.path.join(LEDGER_ROOT, f".{root}.migrating")
        shutil.move(root_current_path, tmp_path)
        os.makedirs(series_path, exist_ok=True)
        shutil.move(tmp_path, os.path.join(series_path, root))

    # 把链上其余仍是扁平存储的成员，搬进同一个系列文件夹
    for name in list_projects():
        if name == root:
            continue
        if series_root(name) != root:
            continue
        current_path = _resolve_existing_path(name)
        if current_path and os.path.dirname(current_path) == LEDGER_ROOT:
            shutil.move(current_path, os.path.join(series_path, name))

    return root


def build_continuity_brief(predecessor_name: str) -> dict:
    """从上一幕的 confirmed 快照提取角色/场景等既有事实，用于新一幕的种子快照。
    只搬运"已经成立的事实"（角色、场景、类型基调），不搬运 want/obstacle——
    新一幕的主线必须由 brain 重新推导，不能直接复用上一幕的。
    """
    pred = load_snapshot(predecessor_name)
    pred_rb = pred.get("rule_based", {})
    carried_characters = []
    for c in pred_rb.get("characters", []):
        carried = dict(c)
        carried["embodies"] = None
        carried["relation_to_throughline"] = "unclear"
        carried_characters.append(carried)
    return {
        "characters": carried_characters,
        "setting": pred_rb.get("setting") or {"location": None, "time_period": None},
        "genre": pred_rb.get("genre"),
        "tone": pred_rb.get("tone"),
        "recap": pred.get("_last_reply", ""),
    }


class ContinuationError(Exception):
    """承接源项目还没收尾，不能作为衔接起点。"""


def create_continuation(new_name: str, predecessor_name: str) -> dict:
    """创建一个承接自 predecessor_name 的新项目，返回种子快照（已落盘）。
    会把 predecessor 所在的整条链（如果还是扁平存储）迁移进同一个系列文件夹，
    新的一幕直接创建在该系列文件夹下。"""
    pred_state = load_snapshot(predecessor_name)["_meta"]["project_state"]
    if pred_state != "confirmed":
        raise ContinuationError(
            f"「{predecessor_name}」还没收尾（当前状态: {pred_state}），不能作为衔接源——"
            f"衔接摘要来自收尾时生成的结局描述，没收尾就没有可用的结局内容。"
        )

    series = _migrate_chain_to_series(predecessor_name)

    brief = build_continuity_brief(predecessor_name)
    snapshot = empty_snapshot()
    snapshot["rule_based"]["characters"] = brief["characters"]
    snapshot["rule_based"]["setting"] = brief["setting"]
    snapshot["rule_based"]["genre"] = brief["genre"]
    snapshot["rule_based"]["tone"] = brief["tone"]
    snapshot["_meta"]["predecessor"] = predecessor_name
    snapshot["_meta"]["continuity_recap"] = brief["recap"]

    new_path = os.path.join(LEDGER_ROOT, series, new_name)
    os.makedirs(new_path, exist_ok=True)
    with open(os.path.join(new_path, "snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return snapshot


def find_successors(project_name: str) -> list:
    """找出所有 predecessor 指向 project_name 的项目（谁承接自它）。"""
    successors = []
    for name in list_projects():
        if name == project_name:
            continue
        snap = load_snapshot(name)
        if snap["_meta"].get("predecessor") == project_name:
            successors.append(name)
    return successors


def find_leaf_descendants(project_name: str) -> list:
    """从 project_name 顺着 successor 一直往下找，返回所有"没有更晚幕次接着写"
    的末端。正常（没有分叉）情况下只有一个，就是这条链最新的一幕；如果曾经从
    同一幕分出过多个分支，会返回多个。用于"选承接对象时，这个是不是已经不是
    最新的了"这类提示——不阻止用户从非最新的幕次续写（分支本身合理），
    只是要让用户清楚自己在做什么。"""
    successors = find_successors(project_name)
    if not successors:
        return [project_name]
    leaves = []
    for s in successors:
        leaves.extend(find_leaf_descendants(s))
    return leaves


def delete_project(project_name: str) -> None:
    """删除整个项目文件夹（快照+流水）。不做任何依赖检查——
    检查/警告是调用方（server.py）的职责，ledger 只负责执行。
    如果删除后它所在的系列文件夹变空了，顺手把这个空文件夹也清掉。"""
    path = _resolve_existing_path(project_name)
    if not path:
        return
    shutil.rmtree(path)

    parent = os.path.dirname(path)
    if parent != LEDGER_ROOT and os.path.isdir(parent) and not os.listdir(parent):
        os.rmdir(parent)


def _drafts_dir(project_name: str) -> str:
    path = os.path.join(project_dir(project_name), "drafts")
    os.makedirs(path, exist_ok=True)
    return path


def save_draft(project_name: str, mode: str, content: str) -> str:
    """存一份写手产出的草稿，文件名带模式和版本号，同一项目可以反复生成
    不同模式/不同版本，互不覆盖。返回写入的文件名（不含目录）。"""
    d = _drafts_dir(project_name)
    existing = [f for f in os.listdir(d) if f.startswith(f"{mode}_v") and f.endswith(".md")]
    versions = []
    for f in existing:
        try:
            versions.append(int(f[len(f"{mode}_v"):-len(".md")]))
        except ValueError:
            continue
    next_version = (max(versions) + 1) if versions else 1
    filename = f"{mode}_v{next_version}.md"
    with open(os.path.join(d, filename), "w", encoding="utf-8") as f:
        f.write(content)
    return filename


def list_drafts(project_name: str) -> list:
    d = _drafts_dir(project_name)
    return sorted(f for f in os.listdir(d) if f.endswith(".md"))


def load_draft(project_name: str, filename: str) -> str:
    path = os.path.join(_drafts_dir(project_name), filename)
    with open(path, encoding="utf-8") as f:
        return f.read()


def latest_draft_filename(project_name: str, mode: str) -> str:
    """project_name 这个项目里，指定 mode 版本号最大的草稿文件名；没有就返回 None。
    "版本号最大 = 最新" 是唯一的规则，续写/事实清单/前端展示全部用这一条，
    不存在别的隐藏逻辑。"""
    prefix, suffix = f"{mode}_v", ".md"
    candidates = [f for f in list_drafts(project_name) if f.startswith(prefix) and f.endswith(suffix)]
    if not candidates:
        return None

    def version_of(f):
        try:
            return int(f[len(prefix):-len(suffix)])
        except ValueError:
            return 0

    return max(candidates, key=version_of)


def latest_draft(project_name: str, mode: str) -> str:
    """project_name 这个项目里，指定 mode 最新版本的草稿全文；没有就返回 None。"""
    filename = latest_draft_filename(project_name, mode)
    return load_draft(project_name, filename) if filename else None


def delete_draft(project_name: str, filename: str) -> None:
    """删除单篇正文。如果删的正好是当前最新版本，连带删掉运行事实清单
    （state 文件本来就是从"最新版本"提炼出来的，最新版本没了，这份清单
    也不再对应任何存在的正文，留着反而会误导下一幕的生成）。如果删的是
    某个更早的历史版本，最新版本没变，事实清单不受影响，不用动。"""
    mode = filename.split("_v", 1)[0] if "_v" in filename else None
    was_latest = mode and latest_draft_filename(project_name, mode) == filename

    path = os.path.join(_drafts_dir(project_name), filename)
    if os.path.exists(path):
        os.remove(path)

    if was_latest:
        state_path = os.path.join(_drafts_dir(project_name), f"{mode}_state.md")
        if os.path.exists(state_path):
            os.remove(state_path)


def chain_from_root(project_name: str) -> list:
    """从系列根一路到 project_name 的完整链条（含两端），按时间顺序排列。"""
    chain = []
    current = project_name
    seen = set()
    while current and current not in seen:
        chain.append(current)
        seen.add(current)
        current = load_snapshot(current)["_meta"].get("predecessor")
    return list(reversed(chain))


def compile_series(project_name: str, mode: str) -> str:
    """把 project_name 所在系列、从根到它自己这一段链条的正文按顺序拼接成一份文档。
    纯字符串拼接，不调用模型——某一幕如果还没生成对应 mode 的正文，会在拼接结果里
    留一句提示，不是报错、也不会跳过导致幕次错位。"""
    chain = chain_from_root(project_name)
    parts = []
    for i, name in enumerate(chain, start=1):
        content = latest_draft(name, mode)
        if content is None:
            mode_label = "小说" if mode == "novel" else "剧本"
            content = f"（这一幕还没有生成{mode_label}正文）"
        parts.append(f"# 第{i}幕：{name}\n\n{content}")
    return "\n\n---\n\n".join(parts)


def save_world_state(project_name: str, mode: str, state_text: str) -> None:
    """存一份某一幕、某模式的"运行事实清单"（跟正文分开存，供后续幕次做事实核对）。"""
    path = os.path.join(_drafts_dir(project_name), f"{mode}_state.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(state_text)


def load_world_state(project_name: str, mode: str) -> str:
    path = os.path.join(_drafts_dir(project_name), f"{mode}_state.md")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def stale_reference_info(project_name: str, mode: str) -> dict:
    """检查 project_name 这一幕（某个 mode）生成时参考的上一幕版本，跟上一幕
    现在的最新版本是否还对得上。返回 {"stale": bool, "generated_from": str|None,
    "predecessor_latest": str|None}。project_name 自己还没生成过 mode 正文、
    或者没有 predecessor，都不算"过期"，直接返回 stale=False。"""
    snapshot = load_snapshot(project_name)
    predecessor = snapshot["_meta"].get("predecessor")
    if not predecessor or latest_draft_filename(project_name, mode) is None:
        return {"stale": False, "generated_from": None, "predecessor_latest": None}

    generated_from = (snapshot["_meta"].get("generated_from") or {}).get(mode)
    predecessor_latest = latest_draft_filename(predecessor, mode)
    return {
        "stale": generated_from != predecessor_latest,
        "generated_from": generated_from,
        "predecessor_latest": predecessor_latest,
    }


def gather_world_state_chain(project_name: str, mode: str, max_acts: int = 5) -> str:
    """从 project_name 的 predecessor 链往回收集最近 max_acts 幕的运行事实清单
    （不含 project_name 自己），按时间顺序拼接。只收集提炼过的摘要，不是原文全文，
    所以链条再长也不会让 token 随幕数线性膨胀——这是跟 previous_excerpt（只传
    最近一幕原文，用于文风）刻意分开设计的两条通道。"""
    chain = chain_from_root(project_name)  # 含 project_name 自己，顺序 root -> ... -> 自己
    prior_acts = chain[:-1][-max_acts:]
    parts = []
    for name in prior_acts:
        state = load_world_state(name, mode)
        if state:
            parts.append(f"【{name}】\n{state}")
    return "\n\n".join(parts) if parts else None


def missing_draft_acts(project_name: str, mode: str) -> list:
    """从系列根到 project_name，按顺序返回还没有 mode 对应正文的幕次名字。"""
    return [name for name in chain_from_root(project_name) if latest_draft(name, mode) is None]


def dependent_drafts(project_name: str, mode: str) -> list:
    """找出所有直接/间接承接自 project_name、且已经生成过 mode 正文的幕次——
    这些幕次的文风/事实参考是基于 project_name 当前这版正文生成的，如果
    project_name 被重新生成，它们参考的就是旧版本了，不再准确。用于重新生成前
    的风险提示，不用于阻止操作本身。"""
    result = []
    for successor in find_successors(project_name):
        if latest_draft(successor, mode) is not None:
            result.append(successor)
        result.extend(dependent_drafts(successor, mode))
    return result
