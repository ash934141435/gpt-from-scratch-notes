"""新版教材的结构、覆盖、截图、依赖、代码索引与导航检查。"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "course"
errors: list[str] = []

CHAPTERS = (
    ("00", "00-学习前准备", "学习前准备", None, None),
    ("01", "01-我们究竟要构建什么", "我们究竟要构建什么", 1, 11),
    ("02", "02-文本如何变成数字", "文本如何变成数字", 12, 17),
    ("03", "03-如何制作训练题目", "如何制作训练题目", 18, 29),
    ("04", "04-第一个Bigram模型", "第一个 Bigram 模型", 30, 39),
    ("05", "05-逐token生成", "逐 token 生成", 40, 47),
    ("06", "06-模型如何学习", "模型如何学习", 48, 55),
    ("07", "07-token为什么需要交流", "为什么 token 需要交流", 56, 61),
    ("08", "08-矩阵乘法与因果Mask", "矩阵乘法与因果 Mask", 62, 76),
    ("09", "09-Embedding与位置", "Embedding 与位置", 77, 85),
    ("10", "10-单头Self-Attention", "单头 Self-Attention", 86, 93),
    ("11", "11-Attention规则与缩放", "Attention 的规则和缩放", 94, 101),
    ("12", "12-Multi-Head-Attention", "Multi-Head Attention", 102, 108),
    ("13", "13-FeedForward-Block与残差", "FeedForward、Block 与残差", 109, 120),
    ("14", "14-LayerNorm与Pre-Norm", "LayerNorm 与 Pre-Norm", 121, 128),
    ("15", "15-完整GPT组装", "完整 GPT 的组装与查漏补缺", 129, 136),
    ("16", "16-Decoder-Only与nanoGPT", "Decoder-Only、原始 Transformer 与 nanoGPT", 137, 148),
    ("17", "17-从预训练到ChatGPT", "从预训练到 ChatGPT 式助手", 149, 164),
)


def read_csv(path: Path, encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline="") as file:
        return list(csv.DictReader(file))


def seconds(clock: str) -> int:
    hour, minute, second = map(int, clock.split(":"))
    return hour * 3600 + minute * 60 + second


def section(text: str, title: str) -> str:
    match = re.search(rf"^## \d+\. {re.escape(title)}\s*$", text, re.MULTILINE)
    if not match:
        return ""
    next_heading = re.search(r"^## \d+\. ", text[match.end() :], re.MULTILINE)
    end = match.end() + next_heading.start() if next_heading else len(text)
    return text[match.end() : end]


# Markdown 语法、本地链接与面向读者的机器路径。
for markdown in ROOT.rglob("*.md"):
    text = markdown.read_text(encoding="utf-8")
    if text.count("```") % 2:
        errors.append(f"unclosed code fence: {markdown.relative_to(ROOT)}")
    for target in re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", text):
        target = target.strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        local = target.split("#", 1)[0]
        if local and not (markdown.parent / local).resolve().exists():
            errors.append(f"broken link: {markdown.relative_to(ROOT)} -> {target}")

reader_markdown = [
    path
    for path in ROOT.rglob("*.md")
    if "sources" not in path.parts and "qa" not in path.parts
]
for markdown in reader_markdown:
    text = markdown.read_text(encoding="utf-8")
    if re.search(r"/Users/[^/]+/|/home/[^/]+/|codex-runtimes", text):
        errors.append(f"machine-specific command path: {markdown.relative_to(ROOT)}")


# 原始视频时间轴、字幕和人工复核仍是事实来源。
timeline = read_csv(ROOT / "qa/timeline.csv")
expected_ids = [f"M{number:03d}" for number in range(1, 165)]
if len(timeline) != 164:
    errors.append(f"expected 164 timeline rows, got {len(timeline)}")
if [row["micro_id"] for row in timeline] != expected_ids:
    errors.append("timeline IDs are not exactly M001-M164 in order")
if timeline and (timeline[0]["start"] != "00:00:00" or timeline[-1]["end"] != "01:56:20"):
    errors.append("timeline does not cover exactly 00:00:00-01:56:20")
for previous, current in zip(timeline, timeline[1:]):
    if seconds(previous["end"]) != seconds(current["start"]):
        errors.append(f"timeline gap/overlap: {previous['micro_id']} -> {current['micro_id']}")

captions = read_csv(ROOT / "qa/subtitle-coverage.csv")
if len(captions) != 2955:
    errors.append(f"expected 2955 caption events, got {len(captions)}")
if {row["micro_id"] for row in captions} != set(expected_ids):
    errors.append("timeline and subtitle coverage use different micro IDs")
pending_captions = sorted(
    {row["micro_id"] for row in captions if row["status"] != "已覆盖并人工复核"}
)
if pending_captions:
    errors.append(f"subtitle review pending: {', '.join(pending_captions)}")

semantic_reviews = read_csv(ROOT / "qa/semantic-review.csv")
review_ids = [row["micro_id"] for row in semantic_reviews]
if len(review_ids) != len(set(review_ids)) or set(review_ids) != set(expected_ids):
    errors.append("semantic review IDs must be unique M001-M164")
invalid_reviews = [
    row["micro_id"]
    for row in semantic_reviews
    if row["status"] != "已人工复核" or not row["notes"].strip()
]
if invalid_reviews:
    errors.append(f"invalid semantic review rows: {', '.join(invalid_reviews)}")


# 新版 M 到学习章节的唯一映射。
learner_map = read_csv(ROOT / "qa/learner-chapter-map.csv")
if [row["micro_id"] for row in learner_map] != expected_ids:
    errors.append("learner chapter map must contain M001-M164 exactly once in order")
if len({row["micro_id"] for row in learner_map}) != 164:
    errors.append("learner chapter map has duplicate micro IDs")

expected_by_id: dict[str, str] = {}
for number, _, _, first, last in CHAPTERS[1:]:
    assert first is not None and last is not None
    for micro_number in range(first, last + 1):
        expected_by_id[f"M{micro_number:03d}"] = number

timeline_by_id = {row["micro_id"]: row for row in timeline}
for row in learner_map:
    micro_id = row["micro_id"]
    if row["learner_chapter"] != expected_by_id.get(micro_id):
        errors.append(f"wrong learner chapter for {micro_id}: {row['learner_chapter']}")
    source = timeline_by_id.get(micro_id)
    if source and (row["start"] != source["start"] or row["end"] != source["end"]):
        errors.append(f"learner map time differs from timeline: {micro_id}")
    if row["old_video_chapter"] != source["chapter"]:
        errors.append(f"learner map lost old chapter provenance: {micro_id}")
    if row["source_mode"] != "video" or not row["boundary_review"].strip():
        errors.append(f"learner map review metadata missing: {micro_id}")


# 18 个新章、条件模板、三层练习和章末完整 M 映射。
actual_course_dirs = sorted(path.name for path in COURSE.iterdir() if path.is_dir())
expected_course_dirs = [chapter[1] for chapter in CHAPTERS]
if actual_course_dirs != expected_course_dirs:
    errors.append(f"course directories mismatch: {actual_course_dirs}")

common_headings = (
    "本章只解决什么问题",
    "学习前检查",
    "不使用术语的直观例子",
    "跟着完成最小代码",
    "每行代码在做什么",
    "Shape 变化卡片",
    "为什么这样设计",
    "常见误解与报错",
    "完整示范",
    "填空模仿",
    "独立小任务",
    "过关标准",
    "暂时不用懂什么",
)

course_texts: dict[str, str] = {}
for number, directory, title, first, last in CHAPTERS:
    path = COURSE / directory / "README.md"
    if not path.exists():
        errors.append(f"missing course chapter: {path.relative_to(ROOT)}")
        continue
    text = path.read_text(encoding="utf-8")
    course_texts[number] = text
    source_mode = "foundation" if number == "00" else "video"
    if f"source_mode={source_mode}" not in text:
        errors.append(f"wrong or missing source_mode in chapter {number}")
    if title not in text.splitlines()[0]:
        errors.append(f"chapter title mismatch: {number}")
    for heading in common_headings:
        if not re.search(rf"^## \d+\. {re.escape(heading)}$", text, re.MULTILINE):
            errors.append(f"chapter {number} missing template heading: {heading}")

    fill_section = section(text, "填空模仿")
    task_section = section(text, "独立小任务")
    if "参考答案" not in fill_section:
        errors.append(f"chapter {number} fill-in exercise has no answer")
    if "参考" not in task_section and "任务通过条件" not in task_section:
        errors.append(f"chapter {number} independent task has no check/answer")

    video_section = section(text, "视频关键片段与画面")
    mapping_section = section(text, "视频时间与 M 映射")
    if number == "00":
        if video_section or mapping_section or re.search(r"M\d{3}", text):
            errors.append("foundation chapter must not contain video evidence or M IDs")
        if re.search(r"!\[[^]]*\]\(", text):
            errors.append("foundation chapter must not require screenshot evidence")
        continue

    if not video_section or not mapping_section:
        errors.append(f"video chapter {number} missing video/mapping conditional sections")
        continue
    if len(re.findall(r"`[^`]*\d{1,2}:\d{2}[^`]*`", video_section)) < 2:
        errors.append(f"video chapter {number} has too few inline key timestamps")
    assert first is not None and last is not None
    expected_chapter_ids = [f"M{value:03d}" for value in range(first, last + 1)]
    written_ids = re.findall(r"M\d{3}", mapping_section)
    if written_ids != expected_chapter_ids:
        errors.append(
            f"chapter {number} mapping mismatch: expected {expected_chapter_ids}, got {written_ids}"
        )


# 术语首次定义证据；状态表必须指向真实正文，不依赖术语表链接。
term_rows = read_csv(ROOT / "qa/term-dependencies.csv")
if not term_rows:
    errors.append("term dependency audit is empty")
for row in term_rows:
    chapter_text = course_texts.get(row["first_chapter"], "")
    if row["definition_evidence"] not in chapter_text:
        errors.append(
            f"term definition evidence missing for {row['term']} in chapter {row['first_chapter']}"
        )
    if row["status"] != "已在首次教学使用处定义":
        errors.append(f"term dependency unresolved: {row['term']}")


# 截图新版双向索引。公开仓库只保留新版正文实际引用的裁剪图。
screenshots = read_csv(ROOT / "qa/screenshot-index.csv")
required_screenshot_fields = {
    "learner_chapter",
    "learner_section",
    "used_in_new_text",
    "evidence_role",
}
if screenshots and not required_screenshot_fields.issubset(screenshots[0]):
    errors.append("screenshot index is missing learner migration fields")

image_paths = [row["image_file"] for row in screenshots]
if len(image_paths) != len(set(image_paths)):
    errors.append("screenshot index contains duplicate image paths")
indexed = {row["image_file"]: row for row in screenshots}
linked_images: dict[str, str] = {}
for number, directory, _, _, _ in CHAPTERS:
    path = COURSE / directory / "README.md"
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    for target in re.findall(r"!\[[^]]*\]\(([^)]+)\)", text):
        resolved = (path.parent / target).resolve().relative_to(ROOT).as_posix()
        linked_images[resolved] = number

for row in screenshots:
    image_file = row["image_file"]
    if not (ROOT / image_file).exists():
        errors.append(f"missing indexed screenshot: {image_file}")
    should_be_used = image_file in linked_images
    if (row.get("used_in_new_text") == "是") != should_be_used:
        errors.append(f"new screenshot usage mismatch: {image_file}")
    if should_be_used:
        if row["cropped"] != "是":
            errors.append(f"new text uses non-crop screenshot: {image_file}")
        if row["learner_chapter"] != linked_images[image_file]:
            errors.append(
                f"screenshot linked from chapter {linked_images[image_file]} but mapped to {row['learner_chapter']}: {image_file}"
            )
        if not row["learner_section"] or not row["evidence_role"]:
            errors.append(f"used screenshot lacks learner metadata: {image_file}")

for image_file in linked_images:
    if image_file not in indexed:
        errors.append(f"course image missing from screenshot index: {image_file}")

chapter_asset_dirs = sorted(ROOT.glob("0[1-9]-*"))
asset_paths = {
    path.relative_to(ROOT).as_posix()
    for directory in chapter_asset_dirs
    for path in directory.glob("assets/**/*.png")
}
for path in sorted(asset_paths - set(image_paths)):
    errors.append(f"asset is missing from screenshot index: {path}")

hashes: dict[str, str] = {}
for target in sorted(asset_paths):
    digest = hashlib.sha256((ROOT / target).read_bytes()).hexdigest()
    if digest in hashes:
        errors.append(f"duplicate screenshot content: {hashes[digest]} == {target}")
    else:
        hashes[digest] = target


# 旧正文只能是导航；根目录只有 course/README 是默认学习入口。
legacy_markdown = (
    list(path / "README.md" for path in sorted(ROOT.glob("0[1-9]-*")))
    + [ROOT / "05-Self-Attention/01-前缀聚合与位置编码.md"]
    + [ROOT / "05-Self-Attention/02-注意力性质与缩放.md"]
    + [ROOT / "07-Transformer-Block/01-LayerNorm与PreNorm.md"]
    + [ROOT / "09-GPT与ChatGPT/01-从预训练到ChatGPT.md"]
    + [ROOT / "09-GPT与ChatGPT/02-课程总结.md"]
)
for path in legacy_markdown:
    text = path.read_text(encoding="utf-8")
    if len(text.splitlines()) > 25 or re.search(r"^## M\d{3}", text, re.MULTILINE):
        errors.append(f"legacy prose was not replaced by navigation: {path.relative_to(ROOT)}")

root_entry = (ROOT / "README.md").read_text(encoding="utf-8")
if "./course/README.md" not in root_entry:
    errors.append("root README does not point to the single course entry")
for redirect in ("00-学习路线.md", "环境准备.md", "GPT从零构建-学习文档.md"):
    if "course/" not in (ROOT / redirect).read_text(encoding="utf-8"):
        errors.append(f"root redirect does not point into course: {redirect}")


# V0-V10 主线、V11 选学与基础报告文件。
versions = read_csv(ROOT / "qa/code-evolution.csv")
if [row["version"] for row in versions] != [f"V{number}" for number in range(12)]:
    errors.append("code evolution does not contain V0-V11 in order")
code_files = sorted(ROOT.glob("[0-9][0-9]-*/code/V*.py")) + sorted(ROOT.glob("capstone/V*.py"))
if len(code_files) != 12:
    errors.append(f"expected 12 runnable V0-V11 files, got {len(code_files)}")
for number in range(12):
    if not any(path.name.startswith(f"V{number}-") for path in code_files):
        errors.append(f"missing runnable V{number} file")
if "--smoke-test" not in (ROOT / "capstone/V11-capstone-gpt.py").read_text(encoding="utf-8"):
    errors.append("V11 does not expose --smoke-test")

required_qa = (
    "cpu-benchmark.md",
    "learner-boundary-review.md",
    "learner-chapter-map.csv",
    "term-dependencies.csv",
)
for name in required_qa:
    if not (ROOT / "qa" / name).exists():
        errors.append(f"missing learner QA artifact: {name}")

for number in range(1, 7):
    if not (ROOT / f"qa/batch-audits/batch-{number}.md").exists():
        errors.append(f"missing batch audit: batch-{number}.md")

if errors:
    raise SystemExit("\n".join(errors))

print(
    f"通过：18 个学习章节、{len(timeline)} 个微片段、"
    f"{len(captions)} 个字幕事件、{len(screenshots)} 张截图、"
    f"{len(code_files)} 个代码快照，所有本地链接均有效"
)
