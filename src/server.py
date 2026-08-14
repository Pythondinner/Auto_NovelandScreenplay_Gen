"""
Web 后端：把 brain/ledger/writer 逻辑包一层 HTTP API 给前端用。
状态机、schema、注入防御、矛盾检测都是同一套核心逻辑，这里只是接口层。
"""

import json
import os

from flask import Flask, jsonify, request, send_from_directory

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


load_env()

app = Flask(__name__, static_folder="../web", static_url_path="")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/projects", methods=["GET"])
def list_projects():
    return jsonify({"projects": ledger.project_summaries()})


@app.route("/api/projects/<name>/continue_from", methods=["POST"])
def continue_from(name):
    data = request.get_json(force=True) or {}
    predecessor = (data.get("predecessor") or "").strip()
    force = bool(data.get("force"))
    if not predecessor:
        return jsonify({"error": "missing predecessor"}), 400
    if predecessor not in ledger.list_projects():
        return jsonify({"error": f"project '{predecessor}' not found"}), 404
    if name in ledger.list_projects():
        return jsonify({"error": f"project '{name}' already exists"}), 409

    # predecessor 已经有幕次接过它了，说明它不是这条链最新的一幕——很可能是
    # 选错了（下拉框/名字容易和系列名搞混）。不阻止（分支本身是合理需求），
    # 但要先确认，不能默默创建出一个用户没意识到的分支。
    existing_successors = ledger.find_successors(predecessor)
    if existing_successors and not force:
        return jsonify({
            "needs_confirmation": True,
            "existing_successors": existing_successors,
            "leaf_suggestions": ledger.find_leaf_descendants(predecessor),
        })

    try:
        snapshot = ledger.create_continuation(name, predecessor)
    except ledger.ContinuationError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(snapshot)


@app.route("/api/projects/<name>", methods=["DELETE"])
def delete_project(name):
    if name not in ledger.list_projects():
        return jsonify({"error": f"project '{name}' not found"}), 404

    force = request.args.get("force") == "true"

    # 不带 force 一律不真的删——只返回依赖信息（哪些幕承接自它、它承接自谁），
    # 前端拿这些信息拼确认提示。任何删除都必须走"先查询、用户确认、再带
    # force=true 调一次"这个两步流程，不存在"第一次调用就直接删掉"的情况。
    if not force:
        return jsonify({
            "needs_confirmation": True,
            "successors": ledger.find_successors(name),
            "predecessor": ledger.load_snapshot(name)["_meta"].get("predecessor"),
        })

    ledger.delete_project(name)
    return jsonify({"deleted": name})


@app.route("/api/projects/<name>", methods=["GET"])
def get_project(name):
    return jsonify(ledger.load_snapshot(name))


@app.route("/api/projects/<name>/log", methods=["GET"])
def get_log(name):
    path = os.path.join(ledger.project_dir(name), "log.jsonl")
    entries = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return jsonify({"log": entries})


@app.route("/api/projects/<name>/message", methods=["POST"])
def send_message(name):
    data = request.get_json(force=True) or {}
    user_input = (data.get("message") or "").strip()
    if not user_input:
        return jsonify({"error": "empty message"}), 400

    snapshot = ledger.load_snapshot(name)
    try:
        snapshot = brain.process_turn(snapshot, user_input)
    except brain.BrainCallError as e:
        return jsonify({"error": f"这轮出错了，没有改动任何内容，可以重试：{e}"}), 502

    ledger.save_snapshot(name, snapshot)
    ledger.append_log(name, "user", user_input)
    ledger.append_log(name, "brain", snapshot.get("_last_reply", ""))

    return jsonify({"snapshot": snapshot, "reply": snapshot.get("_last_reply", "")})


@app.route("/api/projects/<name>/drafts", methods=["GET"])
def list_drafts(name):
    if name not in ledger.list_projects():
        return jsonify({"error": f"project '{name}' not found"}), 404
    return jsonify({
        "drafts": ledger.list_drafts(name),
        "staleness": {
            mode: ledger.stale_reference_info(name, mode)
            for mode in ("novel", "screenplay")
        },
    })


@app.route("/api/projects/<name>/drafts/<filename>", methods=["GET"])
def get_draft(name, filename):
    if name not in ledger.list_projects():
        return jsonify({"error": f"project '{name}' not found"}), 404
    try:
        content = ledger.load_draft(name, filename)
    except FileNotFoundError:
        return jsonify({"error": f"draft '{filename}' not found"}), 404
    return jsonify({"filename": filename, "content": content})


@app.route("/api/projects/<name>/drafts/<filename>", methods=["DELETE"])
def delete_draft(name, filename):
    if name not in ledger.list_projects():
        return jsonify({"error": f"project '{name}' not found"}), 404
    if filename not in ledger.list_drafts(name):
        return jsonify({"error": f"draft '{filename}' not found"}), 404
    ledger.delete_draft(name, filename)
    return jsonify({"deleted": filename})


@app.route("/api/projects/<name>/write", methods=["POST"])
def write_draft(name):
    if name not in ledger.list_projects():
        return jsonify({"error": f"project '{name}' not found"}), 404

    snapshot = ledger.load_snapshot(name)
    if snapshot["_meta"]["project_state"] != "confirmed":
        return jsonify({"error": "项目还没收尾，不能生成正文"}), 400

    data = request.get_json(force=True) or {}
    mode = data.get("mode")
    if mode not in ("novel", "screenplay"):
        return jsonify({"error": "mode 必须是 novel 或 screenplay"}), 400

    force = bool(data.get("force"))

    # 已经有正文的情况下，不带 force 一律不真的生成——先返回依赖信息，
    # 前端拼确认提示，用户确认后带 force=true 再调一次，跟删除项目走的
    # 是同一个两步流程。没有已存在正文的情况下不需要这道确认。
    if not force and ledger.latest_draft(name, mode) is not None:
        return jsonify({
            "needs_confirmation": True,
            "dependents": ledger.dependent_drafts(name, mode),
        })

    predecessor = snapshot["_meta"].get("predecessor")
    previous_excerpt_exists = predecessor and ledger.latest_draft(predecessor, mode) is not None

    chapter_reports = []
    try:
        filename, content = writer.write_and_save_one(name, mode, on_chapter=chapter_reports.append)
    except writer.deepseek_client.ApiCallError as e:
        return jsonify({"error": f"生成失败，没有写入任何文件，可以重试：{e}"}), 502

    return jsonify({
        "filename": filename,
        "content": content,
        "continued_from_draft": bool(previous_excerpt_exists),
        "chapters": chapter_reports,  # 只有一段（未分章）时是长度为1的列表，前端据此判断要不要显示分章信息
    })


@app.route("/api/projects/<name>/write_series", methods=["POST"])
def write_series(name):
    if name not in ledger.list_projects():
        return jsonify({"error": f"project '{name}' not found"}), 404

    data = request.get_json(force=True) or {}
    mode = data.get("mode")
    if mode not in ("novel", "screenplay"):
        return jsonify({"error": "mode 必须是 novel 或 screenplay"}), 400

    try:
        results = writer.write_missing_acts_in_series(name, mode)
    except writer.deepseek_client.ApiCallError as e:
        return jsonify({"error": f"批量生成中途失败，已生成的部分已保留，可以重新调用补齐剩余部分：{e}"}), 502

    return jsonify({"results": results})


@app.route("/api/projects/<name>/compile", methods=["GET"])
def compile_series(name):
    if name not in ledger.list_projects():
        return jsonify({"error": f"project '{name}' not found"}), 404
    mode = request.args.get("mode")
    if mode not in ("novel", "screenplay"):
        return jsonify({"error": "mode 必须是 novel 或 screenplay"}), 400
    content = ledger.compile_series(name, mode)
    return jsonify({"content": content, "chain": ledger.chain_from_root(name)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5057, debug=True)
