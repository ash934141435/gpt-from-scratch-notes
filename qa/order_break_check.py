# -*- coding: utf-8 -*-
"""新版 18 章顺序断裂扫描：找出“冒号承诺后被标题或图片截断”。

检查项：
1. promise-break：正文行以全角冒号"："结尾（承诺后面紧跟内容），
   但在兑现内容（列表、代码块、普通段落）之前先出现了小节标题或截图。
用法：python qa/order_break_check.py
输出：控制台摘要 + qa/order-break-report.md
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "qa" / "order-break-report.md"

CHAPTER_DIRS = [
    "course/00-学习前准备",
    "course/01-我们究竟要构建什么",
    "course/02-文本如何变成数字",
    "course/03-如何制作训练题目",
    "course/04-第一个Bigram模型",
    "course/05-逐token生成",
    "course/06-模型如何学习",
    "course/07-token为什么需要交流",
    "course/08-矩阵乘法与因果Mask",
    "course/09-Embedding与位置",
    "course/10-单头Self-Attention",
    "course/11-Attention规则与缩放",
    "course/12-Multi-Head-Attention",
    "course/13-FeedForward-Block与残差",
    "course/14-LayerNorm与Pre-Norm",
    "course/15-完整GPT组装",
    "course/16-Decoder-Only与nanoGPT",
    "course/17-从预训练到ChatGPT",
]

# 迁移前的截图元信息和迁移后的标准图注都属于图片附属内容
IMG_META_RE = re.compile(
    r"^(?:\*\*(视频时间|重点看|它说明)：|\*图：.+（原视频 M\d{3}，\d{2}:\d{2}:\d{2}）\*$)"
)
IMG_RE = re.compile(r"^!\[.*\]\(.*\)\s*$")
HEADING_RE = re.compile(r"^#{2,4}\s")
PROMISE_MAX_LEN = 80  # 承诺行一般很短；过长的是普通叙述，跳过以降低误报


def is_promise_line(line: str) -> bool:
    s = line.strip()
    if not s.endswith("："):
        return False
    if len(s) > PROMISE_MAX_LEN:
        return False
    if s.startswith(("-", "*", ">", "|", "**")):  # 列表/引用/表格/加粗元信息
        return False
    if re.match(r"^\d+\.", s):  # 有序列表项
        return False
    return True


def scan_file(path: Path):
    """返回 [(行号, 承诺行, 插入物描述)] 列表。"""
    findings = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_fence = False
    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            i += 1
            continue
        if in_fence or not is_promise_line(raw):
            i += 1
            continue

        # 向前看：跳过空行，找第一个"实质内容"
        j = i + 1
        blocker = None
        while j < len(lines):
            t = lines[j].strip()
            if not t:
                j += 1
                continue
            if t.startswith("```"):
                # 代码块 = 兑现（除非在到达前有标题/图片，已提前记录）
                break
            if HEADING_RE.match(t):
                blocker = f"第 {j + 1} 行小节标题「{t}」"
                break
            if IMG_RE.match(t) or IMG_META_RE.match(t):
                blocker = f"第 {j + 1} 行截图或其元信息"
                break
            # 普通文本行 / 列表项 = 承诺已兑现
            break
        if blocker:
            findings.append((i + 1, s, blocker))
        i += 1
    return findings


def main():
    report_lines = ["# 顺序断裂扫描报告", ""]
    total_breaks = 0

    for d in CHAPTER_DIRS:
        chapter = ROOT / d
        if not chapter.is_dir():
            print(f"[跳过] 目录不存在：{d}")
            continue
        report_lines.append(f"## {d}")
        report_lines.append("")

        chapter_breaks = []
        for md in sorted(chapter.glob("*.md")):
            for lineno, promise, blocker in scan_file(md):
                chapter_breaks.append((md.name, lineno, promise, blocker))

        if chapter_breaks:
            total_breaks += len(chapter_breaks)
            report_lines.append("| 文件 | 行号 | 承诺行 | 插入物 |")
            report_lines.append("|---|---:|---|---|")
            for name, lineno, promise, blocker in chapter_breaks:
                report_lines.append(f"| {name} | {lineno} | {promise} | {blocker} |")
            report_lines.append("")
        else:
            report_lines.append("- 未发现问题")
            report_lines.append("")

    summary = f"共发现 {total_breaks} 处承诺截断。"
    report_lines.insert(2, summary)
    report_lines.insert(3, "")
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(summary)
    print(f"报告已写入：{REPORT}")
    return 0 if total_breaks == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
