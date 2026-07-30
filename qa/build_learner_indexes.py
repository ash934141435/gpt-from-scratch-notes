"""生成新版学习章节映射，并迁移截图的新版使用状态。"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHAPTERS = (
    (1, "01", "我们究竟要构建什么", 1, 11),
    (2, "02", "文本如何变成数字", 12, 17),
    (3, "03", "如何制作训练题目", 18, 29),
    (4, "04", "第一个 Bigram 模型", 30, 39),
    (5, "05", "逐 token 生成", 40, 47),
    (6, "06", "模型如何学习", 48, 55),
    (7, "07", "为什么 token 需要交流", 56, 61),
    (8, "08", "矩阵乘法与因果 Mask", 62, 76),
    (9, "09", "Embedding 与位置", 77, 85),
    (10, "10", "单头 Self-Attention", 86, 93),
    (11, "11", "Attention 的规则和缩放", 94, 101),
    (12, "12", "Multi-Head Attention", 102, 108),
    (13, "13", "FeedForward、Block 与残差", 109, 120),
    (14, "14", "LayerNorm 与 Pre-Norm", 121, 128),
    (15, "15", "完整 GPT 的组装与查漏补缺", 129, 136),
    (16, "16", "Decoder-Only、原始 Transformer 与 nanoGPT", 137, 148),
    (17, "17", "从预训练到 ChatGPT 式助手", 149, 164),
)


def chapter_for(micro_id: str) -> tuple[int, str, str, int, int]:
    number = int(micro_id[1:])
    return next(chapter for chapter in CHAPTERS if chapter[3] <= number <= chapter[4])


def read_csv(path: Path, encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_chapter_map() -> None:
    timeline = read_csv(ROOT / "qa/timeline.csv")
    rows: list[dict[str, str]] = []
    for source in timeline:
        _, learner_number, title, first, last = chapter_for(source["micro_id"])
        micro_number = int(source["micro_id"][1:])
        boundary = micro_number in (first, last)
        rows.append(
            {
                "micro_id": source["micro_id"],
                "start": source["start"],
                "end": source["end"],
                "old_video_chapter": source["chapter"],
                "learner_chapter": learner_number,
                "learner_title": title,
                "source_mode": "video",
                "boundary_review": "增量核验" if boundary else "复用现有人工QA",
                "review_basis": (
                    "timeline+subtitle-coverage+semantic-review；检查滚动字幕跨界归属"
                    if boundary
                    else "timeline+semantic-review"
                ),
            }
        )
    write_csv(
        ROOT / "qa/learner-chapter-map.csv",
        [
            "micro_id",
            "start",
            "end",
            "old_video_chapter",
            "learner_chapter",
            "learner_title",
            "source_mode",
            "boundary_review",
            "review_basis",
        ],
        rows,
    )


def linked_course_images() -> set[str]:
    linked: set[str] = set()
    for markdown in (ROOT / "course").rglob("README.md"):
        text = markdown.read_text(encoding="utf-8")
        for target in re.findall(r"!\[[^]]*\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://")):
                continue
            resolved = (markdown.parent / target).resolve()
            linked.add(resolved.relative_to(ROOT).as_posix())
    return linked


def migrate_screenshot_index() -> None:
    path = ROOT / "qa/screenshot-index.csv"
    rows = read_csv(path, encoding="utf-8-sig")
    linked = linked_course_images()
    migrated: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        _, number, _, _, _ = chapter_for(row["micro_id"])
        used = row["image_file"] in linked
        if not used:
            continue
        row["learner_chapter"] = number
        row["learner_section"] = "视频关键片段与画面"
        row["used_in_new_text"] = "是"
        row["evidence_role"] = "正文视频证据"
        migrated.append(row)

    additions = ["learner_chapter", "learner_section", "used_in_new_text", "evidence_role"]
    original = [field for field in rows[0] if field not in additions]
    write_csv(path, original + additions, migrated)


if __name__ == "__main__":
    build_chapter_map()
    migrate_screenshot_index()
    print("已重建 learner-chapter-map.csv，并更新公开截图索引")
