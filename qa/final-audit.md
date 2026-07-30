# 零机器学习基础版教材最终审计

审计日期：2026-07-29  
审计范围：18 章课程正文、根入口、旧九章导航、视频/字幕映射、截图索引、术语依赖、冷读记录和 V0–V11。

## 结论

九章逐微段正文已重构为第 00–17 章。第 00 章补足环境、张量、Shape、Python 类和报错阅读；第 01–17 章连续覆盖 M001–M164。根目录默认学习入口统一为 `course/README.md`，旧章节只保留短导航，代码和媒体未搬迁。

自动结构、字幕、映射、截图、链接、冷读记录和代码验收均通过。尚未完成的非机器项是 **1–2 名真实零基础读者的复测**；当前冷读结果明确标为受限上下文大模型代理，不能冒充真人用时与卡点反馈。

## 验收结果

| 项目 | 结果 | 证据 |
|---|---:|---|
| 新课程结构 | 通过 | `course/README.md` + 18 个章节 README，共 3378 行正文 |
| 基础章豁免 | 通过 | 第 00 章 `source_mode=foundation`，无 M、视频或截图要求 |
| 视频章模板 | 通过 | 第 01–17 章均含段内关键时间、15 个栏目和章末完整映射 |
| M 唯一映射 | 通过 | `learner-chapter-map.csv` 中 M001–M164 唯一、连续并保留旧来源字段 |
| 字幕覆盖 | 通过 | 2955 条事件继续标记为已覆盖并人工复核 |
| 增量边界核验 | 通过 | 16 个新边界均落在既有 M 边界；滚动字幕保持单一语义归属 |
| 三层练习 | 通过 | 每章有完整示范、填空模仿、独立任务、答案/参考和过关标准 |
| 术语依赖 | 通过 | `term-dependencies.csv` 的首次定义证据均能在对应正文找到 |
| 截图迁移 | 通过 | 公开仓库仅保留新版正文使用的 17 张裁剪证据，逐章双向可追溯 |
| 旧入口处置 | 通过 | 三个根旧入口与旧九章正文均改为新版导航 |
| 冷读代理 | 通过 | 18 章均有规格和结果，全部达到 5/5 + 独立任务，无后文依赖 |
| 真实读者冷读 | 待补 | 代理筛查不能替代真实读者停顿、误解和用时记录 |
| 代码验收 | 通过 | V0–V10 默认快速配置 + V11 `--smoke-test` 全部成功 |
| CPU 时间 | 通过 | 12 个版本共 19.85 秒；单个最长 4.94 秒 |
| 学习时间 | 已更新 | 课程入口按各章合计标 35.5–50.5 小时 |
| 许可 | 已说明边界 | `THIRD_PARTY_NOTICES.md` 已区分原创内容与第三方素材；统一开源许可证仍由权利人决定 |

## QA 复用与新增范围

本轮没有重新从头观看整部视频。以下旧资产继续作为事实输入：

- `timeline.csv`、`subtitle-coverage.csv`、`semantic-review.csv`；
- `chapter-01-review.md` 至 `chapter-09-review.md`；
- `full-timeline-review.md`、`order-break-report.md`；
- 原 `screenshot-index.csv` 的时间、M 编号、裁剪与来源字段。

新增核验只处理新章边界、滚动字幕归属和正文图片是否与新章节 M 范围一致。细节见 `learner-boundary-review.md`。没有发现证据冲突，因此未扩大为全视频重看。

## 自动检查

```bash
python qa/order_break_check.py
python qa/check.py
python qa/check_cold_read.py
```

结果：

```text
共发现 0 处承诺截断。
OK: 18 learner chapters, 164 micro-segments, 2955 caption events,
310 screenshots, 12 code snapshots, all local links valid
OK: 18 cold-read specs and results; every chapter passed >=4/5 plus task
```

`qa/check.py` 以 `source_mode` 条件处理第 00 章，不会因无视频而误报。截图验收只读取 `used_in_new_text`，不会沿用旧 `used_in_text` 冒充新版通过。

## 代码实跑与 CPU 基准

环境：Python 3.10.12、PyTorch 2.13.0+cpu、NumPy 2.2.6。

```bash
python qa/run_code_snapshots.py
```

关键结果：

- V0 编码/解码往返；V1 生成正确 `[B,T]` x/y；
- V2 完成 Bigram loss 和生成；V3 验证 loss 从 4.7148 降至 2.5877；
- V4 三种因果前缀平均等价；V5/V6 单头和多头 Shape 通过；
- V7 验证 FFN 不混合时间；V8 验证恒等残差与梯度直通；
- V9 完整 pre-norm Block Shape 和梯度有限；
- V10 验证 loss 从 4.330 降至 3.908，并生成 81 个 token；
- V11 只跑两步 smoke：固定评估、perplexity、checkpoint 保存/新实例加载、temperature/top-k 采样均通过。

详细逐版本时间见 `cpu-benchmark.md`。本地完整套件 19.85 秒，低于约 3 分钟目标；V11 没有在自动验收中运行默认 200 步训练。

## 教学结构核对

Attention 被拆成六级连续台阶：

1. 第 07 章：为什么要通信与循环前缀平均；
2. 第 08 章：矩阵乘法、下三角、mask 与 softmax；
3. 第 09 章：token/position embedding 和 V/T/C；
4. 第 10 章：单头 Q/K/V；
5. 第 11 章：有向图、mask/位置/内容分离和缩放；
6. 第 12 章：ModuleList、多头并行与拼接。

第 15 章的新增核心概念经审计只有 Dropout 和规模/资源成本。Embedding、Block、final norm、lm_head、loss、训练和生成全部指回前章，不再在综合章重新起课。

## 冷读解释

`qa/cold-read/` 为每章保存五题、独立任务、允许范围、禁止后续术语和结果。当前测试者类型全部为 `llm_proxy`，只允许读取当前及前文章节。自动脚本只检查结果文件、4/5 门槛、任务通过和无后文依赖，不负责生成或主观评分答案。

下一步真实读者复测时，应优先记录第 00–03 章的完成时间、错误、未解释词和练习卡点；若真实反馈与代理不同，以真实反馈推动修订，并保留两类结果的来源标识。

## 剩余边界

- V10 是必学终点；V11、checkpoint、perplexity、temperature、top-k 与现代架构是选学。
- 第 17 章的 SFT/奖励模型/PPO 是带 2022 来源的历史概念地图，不代表 2026 年所有系统固定流程。
- 媒体与上游代码许可不自动等同；顶层 LICENSE 不在本次重构中选择。
- 当前目录没有 Git 元数据，本轮无法产生提交差异报告；所有可复核结果以文件和 QA 输出为准。
