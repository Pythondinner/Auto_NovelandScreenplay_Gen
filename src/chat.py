"""
交互层的终端入口：多轮对话，接 ledger 持久化 + brain 状态机。
支持完整流程（empty -> candidates_pending -> collecting -> confirmed）。

用法：
    python src/chat.py [项目名] [承接自的项目名]

项目名省略时默认 "default"；同一个项目名下次运行会从上次的 ledger 快照继续。
第二个参数可选：如果这是个全新项目名，且指定了承接自哪个已有项目，会自动继承
该项目的角色/场景设定（衔接摘要机制），但 want/obstacle 需要重新确立。
输入 exit / quit 退出。输入 状态 可以随时查看当前 snapshot（不消耗模型调用）。
"""

import json
import os
import sys

import brain
import ledger
import writer


def load_env(path=".env"):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def print_snapshot(snapshot: dict) -> None:
    view = {k: v for k, v in snapshot.items() if not k.startswith("_")}
    view["_meta"] = snapshot["_meta"]
    print(json.dumps(view, ensure_ascii=False, indent=2))


def choose_project() -> str:
    """未指定项目名时，展示已有项目列表（含承接关系），让用户选择继续还是开一个新的。"""
    summaries = ledger.project_summaries()
    if summaries:
        print("已有的剧本项目：")
        for s in summaries:
            tail = f" ← 承接自 {s['predecessor']}" if s["predecessor"] else ""
            print(f"  - {s['name']}（状态: {s['state']}{tail}）")
        print()
        choice = input("输入已有项目名继续，或直接输入一个新名字开始新剧本：").strip()
    else:
        choice = input("给这个新剧本项目起个名字（直接回车用默认名 'default'）：").strip()
    return choice or "default"


def maybe_link_predecessor(project_name: str, predecessor_arg: str = None) -> None:
    """如果 project_name 是全新项目，询问（或使用命令行参数）是否承接自某个已有项目。"""
    existing = ledger.list_projects()
    if project_name in existing:
        return  # 已存在，不是新项目，不需要问
    if not existing:
        return  # 没有其他项目可承接

    predecessor = predecessor_arg
    if not predecessor:
        choice = input(
            f'「{project_name}」是新项目。要承接自某个已有项目吗？'
            f'输入项目名，或直接回车表示全新故事：'
        ).strip()
        predecessor = choice or None

    if not predecessor:
        return
    if predecessor not in existing:
        print(f"没找到项目「{predecessor}」，将创建全新故事，不做承接。\n")
        return

    # predecessor 已经有幕次接过它了，很可能选错了（想要的是最新一幕，
    # 不是这个）——不阻止（分支本身合理），但要先提醒。
    existing_successors = ledger.find_successors(predecessor)
    if existing_successors:
        leaves = ledger.find_leaf_descendants(predecessor)
        print(
            f"注意：「{predecessor}」已经有幕次承接自它了：{'、'.join(existing_successors)}。"
            f"这条链目前最新的一幕是：{'、'.join(leaves)}。"
        )
        if input(f"确定要从「{predecessor}」（不是最新的）另开一个分支吗？输入 是 确认，其他任意键取消续写：").strip() != "是":
            print("已取消续写，将创建全新故事，不做承接。\n")
            return

    try:
        ledger.create_continuation(project_name, predecessor)
        print(f"已从「{predecessor}」的结局继承角色/场景设定，want/obstacle 需要重新聊出来。\n")
    except ledger.ContinuationError as e:
        print(f"{e}\n将创建全新故事，不做承接。\n")


MODE_MAP = {"小说": "novel", "novel": "novel", "剧本": "screenplay", "screenplay": "screenplay"}
MODE_LABEL = {"novel": "小说", "screenplay": "剧本"}


def offer_to_write(project_name: str, snapshot: dict) -> None:
    """项目收尾后，问用户要不要现在生成正文，支持连续生成多个模式/多个版本、
    批量补齐系列里还没写的幕次。"""
    while True:
        choice = input(
            "\n要生成正文吗？输入 小说 或 剧本 选择格式；"
            "输入 批量小说 或 批量剧本 补齐这个系列里还没生成的幕次；"
            "输入 删除<文件名> 删掉某个版本（比如 删除novel_v2.md）；"
            "直接回车跳过：\n> "
        ).strip()
        if not choice:
            return

        if choice.startswith("删除"):
            filename = choice[2:].strip()
            if filename not in ledger.list_drafts(project_name):
                print(f"没找到「{filename}」，当前有：{'、'.join(ledger.list_drafts(project_name))}")
                continue
            ledger.delete_draft(project_name, filename)
            print(f"已删除 {filename}（如果它是当时最新版本，对应的运行事实清单也一并清掉了）")
            continue

        if choice.startswith("批量"):
            mode = MODE_MAP.get(choice[2:])
            if not mode:
                print(f"没识别出「{choice}」，请输入 批量小说 或 批量剧本。")
                continue
            print(f"正在批量补齐{MODE_LABEL[mode]}正文（按顺序逐幕生成，可能需要几分钟）…")
            try:
                results = writer.write_missing_acts_in_series(
                    project_name, mode,
                    on_progress=lambda name, i, total: print(f"  [{i}/{total}] {name} …"),
                )
            except writer.deepseek_client.ApiCallError as e:
                print(f"（批量生成中途失败，已生成的部分已保留，可以重新执行批量生成补齐剩余部分：{e}）")
                continue
            for r in results:
                if r["status"] == "written":
                    print(f"  ✓ {r['name']}: 已生成 {r['filename']}")
                elif r["status"] == "skipped_exists":
                    print(f"  - {r['name']}: 已有正文，跳过")
                else:
                    print(f"  ✗ {r['name']}: 还没收尾，停在这里，后面的幕次没有处理")
            continue

        mode = MODE_MAP.get(choice)
        if not mode:
            print(f"没识别出「{choice}」，请输入 小说、剧本、批量小说 或 批量剧本。")
            continue

        if ledger.latest_draft(project_name, mode) is not None:
            dependents = ledger.dependent_drafts(project_name, mode)
            warn = f"「{project_name}」已经有{MODE_LABEL[mode]}正文了，重新生成会存成新版本。"
            if dependents:
                warn += (
                    f"\n⚠ 已经有幕次承接自它并参考它生成过正文：{'、'.join(dependents)}。"
                    f"重新生成这一幕后，它们参考的还是旧版本，可能出现新的不一致，"
                    f"建议重新生成完这一幕后也考虑重新生成它们。"
                )
            print(warn)
            if input("确定要重新生成吗？输入 是 确认，其他任意键取消：").strip() != "是":
                continue

        predecessor = snapshot["_meta"].get("predecessor")
        if predecessor and ledger.latest_draft(predecessor, mode) is None:
            print(f"（提示：上一幕「{predecessor}」还没生成过{MODE_LABEL[mode]}正文，这一幕会独立生成，文风可能接不上）")

        print(f"正在生成{MODE_LABEL[mode]}正文，可能需要一点时间…")

        def on_chapter(report):
            note = "（这一段被截断了，不是正常写完的）" if report["finish_reason"] == "length" else ""
            if report["total"] > 1:
                print(f"  [{report['index']}/{report['total']}] 已写 {report['char_count']} 字 {note}")

        try:
            filename, content = writer.write_and_save_one(project_name, mode, on_chapter=on_chapter)
        except writer.deepseek_client.ApiCallError as e:
            print(f"（生成失败，没有写入任何文件，可以重试：{e}）")
            continue

        print(f"\n已生成并存入 ledger/…/drafts/{filename}，预览：\n")
        preview = content if len(content) <= 600 else content[:600] + "…（后面还有）"
        print(preview)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    load_env()

    predecessor_arg = sys.argv[2] if len(sys.argv) > 2 else None
    project_name = sys.argv[1] if len(sys.argv) > 1 else choose_project()
    maybe_link_predecessor(project_name, predecessor_arg)

    snapshot = ledger.load_snapshot(project_name)

    print(f"\n=== 项目: {project_name} | 当前状态: {snapshot['_meta']['project_state']} ===")
    print("（输入 exit/quit 退出，输入 状态 查看当前完整 snapshot）\n")

    state = snapshot["_meta"]["project_state"]
    if state == "confirmed":
        print(
            "这个项目已经收尾了。输入 状态 可以查看最终内容；"
            "想开新剧本的话退出后用新名字运行：python src/chat.py 新名字\n"
        )
        offer_to_write(project_name, snapshot)
        return
    elif state == "empty" and snapshot["_meta"].get("predecessor"):
        print(f"brain: 接着「{snapshot['_meta']['predecessor']}」的结局，这一幕想往哪个方向走？")
    elif state == "empty" and not snapshot.get("raw_user_request"):
        print("brain: 想聊聊你想写个什么样的剧本吗？")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input == "状态":
            print_snapshot(snapshot)
            continue

        try:
            snapshot = brain.process_turn(snapshot, user_input)
        except brain.BrainCallError as e:
            print(f"\n（这轮出错了，没有改动任何内容，可以重新输入试一次：{e}）\n")
            continue
        ledger.save_snapshot(project_name, snapshot)
        ledger.append_log(project_name, "user", user_input)
        ledger.append_log(project_name, "brain", snapshot.get("_last_reply", ""))

        state = snapshot["_meta"]["project_state"]
        print(f"\nbrain [{state}]: {snapshot.get('_last_reply', '')}\n")

        if snapshot["_meta"].get("flagged_injection"):
            print(f"  (系统提示：本轮检测到疑似注入内容，已标记：{snapshot['_meta']['flagged_snippets']})\n")
        if snapshot["_meta"].get("conflicts_flagged"):
            print(f"  (系统提示：检测到潜在矛盾：{snapshot['_meta']['conflicts_flagged']})\n")

        if state == "confirmed":
            offer_to_write(project_name, snapshot)
            break


if __name__ == "__main__":
    main()
