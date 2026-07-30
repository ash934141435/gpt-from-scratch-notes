"""检查每章冷读规格和结果记录，不生成或主观评分答案。"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLD_READ = ROOT / "qa/cold-read"
errors: list[str] = []

for number in range(18):
    chapter = f"{number:02d}"
    spec_path = COLD_READ / f"chapter-{chapter}.json"
    result_path = COLD_READ / f"chapter-{chapter}-result.json"
    if not spec_path.exists():
        errors.append(f"missing cold-read spec: chapter-{chapter}.json")
        continue
    if not result_path.exists():
        errors.append(f"missing cold-read result: chapter-{chapter}-result.json")
        continue

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if spec.get("chapter") != chapter or result.get("chapter") != chapter:
        errors.append(f"chapter ID mismatch in cold-read files: {chapter}")
    questions = spec.get("questions", [])
    if len(questions) != 5 or any(not question.strip() for question in questions):
        errors.append(f"chapter {chapter} must define five cold-read questions")
    if not spec.get("task") or not spec.get("allowed_chapters"):
        errors.append(f"chapter {chapter} cold-read scope/task missing")
    if spec.get("allowed_chapters", [])[-1:] != [chapter]:
        errors.append(f"chapter {chapter} allowed scope must end at current chapter")
    rubric = spec.get("rubric", {})
    if rubric.get("minimum_questions") != 4 or rubric.get("task_required") is not True:
        errors.append(f"chapter {chapter} rubric does not enforce 4/5 plus task")

    passed = result.get("question_results", [])
    if len(passed) != 5 or any(type(value) is not bool for value in passed):
        errors.append(f"chapter {chapter} result must contain five booleans")
    elif sum(passed) < 4:
        errors.append(f"chapter {chapter} passed fewer than four questions")
    if result.get("task_passed") is not True:
        errors.append(f"chapter {chapter} independent task did not pass")
    if result.get("later_dependency") is not False:
        errors.append(f"chapter {chapter} depends on a later chapter")
    if result.get("tester_type") not in {"llm_proxy", "real_reader"}:
        errors.append(f"chapter {chapter} has invalid tester type")
    if result.get("tester_type") == "llm_proxy" and result.get("scope_enforced") is not True:
        errors.append(f"chapter {chapter} proxy scope was not enforced")
    unresolved = result.get("unexplained_terms", [])
    resolved = result.get("resolved_or_optional_terms", [])
    if any(term not in resolved for term in unresolved):
        errors.append(f"chapter {chapter} contains unresolved unexplained terms")

if errors:
    raise SystemExit("\n".join(errors))

print("通过：18 章均有冷读规范和结果，每章评分不低于 4/5 且任务通过")
