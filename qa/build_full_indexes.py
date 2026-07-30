"""生成冻结模板后的全视频语义时间轴与字幕覆盖表。"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sec(clock: str) -> int:
    hour, minute, second = map(int, clock.split(":"))
    return hour * 3600 + minute * 60 + second


# 每项：结束时间、标题、内容类型、章节。
# 上一项结束时间自动成为下一项开始时间，避免手工制造时间缺口。
SEGMENTS = [
    # 01 课程介绍
    ("00:00:36", "观察 ChatGPT 从左到右生成文本", "演示", "01-课程介绍与数据准备"),
    ("00:01:03", "同一提示为什么会得到不同回答", "概念", "01-课程介绍与数据准备"),
    ("00:01:38", "从趣味案例认识文本任务的多样性", "演示", "01-课程介绍与数据准备"),
    ("00:02:07", "语言模型就是序列补全器", "概念", "01-课程介绍与数据准备"),
    ("00:02:44", "ChatGPT 底层为什么是 Transformer", "概念", "01-课程介绍与数据准备"),
    ("00:03:13", "机器翻译论文如何成为通用架构", "背景", "01-课程介绍与数据准备"),
    ("00:03:52", "本课不复现 ChatGPT 而是学习核心机制", "目标", "01-课程介绍与数据准备"),
    ("00:04:16", "选择 tiny Shakespeare 作为玩具数据集", "数据", "01-课程介绍与数据准备"),
    ("00:05:21", "字符级下一个字符预测与生成预览", "任务与演示", "01-课程介绍与数据准备"),
    ("00:06:25", "nanoGPT 的两个核心文件与 GPT-2 验证", "代码背景", "01-课程介绍与数据准备"),
    ("00:07:29", "从空文件构建 Transformer 与前置要求", "学习路线", "01-课程介绍与数据准备"),
    # 02 字符编码
    ("00:08:34", "在 Colab 下载、读取并检查原始文本", "代码与运行结果", "02-字符编码与训练数据"),
    ("00:09:36", "用 set 与 sorted 建立 65 字符词表", "代码与数据", "02-字符编码与训练数据"),
    ("00:10:20", "Tokenizer 为什么要把字符串变成整数", "概念与代码", "02-字符编码与训练数据"),
    ("00:11:43", "建立 stoi 与 itos 双向查找表", "代码", "02-字符编码与训练数据"),
    ("00:12:05", "字符级与 subword 分词器有什么不同", "概念对比", "02-字符编码与训练数据"),
    ("00:12:45", "词表大小与序列长度之间的权衡", "设计权衡", "02-字符编码与训练数据"),
    ("00:13:48", "把全文编码为 tensor 并划分训练验证集", "代码与数据", "02-字符编码与训练数据"),
    # 训练样本与 batch
    ("00:14:50", "为什么不能一次喂入全文及 block_size 的作用", "概念与参数", "02-字符编码与训练数据"),
    ("00:15:26", "为什么抽取 block_size 加一的字符", "数据构造", "02-字符编码与训练数据"),
    ("00:15:52", "一个长度九的片段如何打包八个样本", "样本推导", "02-字符编码与训练数据"),
    ("00:16:30", "构造偏移一位的输入 x 与目标 y", "代码", "02-字符编码与训练数据"),
    ("00:16:56", "逐位置打印 context 与 target", "运行结果", "02-字符编码与训练数据"),
    ("00:17:32", "训练不同长度上下文的两个原因", "设计解释", "02-字符编码与训练数据"),
    ("00:17:58", "引入 batch 维并保持样本隔离", "概念与shape", "02-字符编码与训练数据"),
    ("00:18:44", "get_batch 随机选择起始位置", "代码", "02-字符编码与训练数据"),
    ("00:19:28", "stack 如何得到 B×T 输入和目标", "代码与shape", "02-字符编码与训练数据"),
    ("00:20:10", "打印 batch 并识别 32 个训练位置", "运行结果", "02-字符编码与训练数据"),
    ("00:22:16", "batch 让多条样本共同估计梯度但样本之间不通信", "概念回顾", "02-字符编码与训练数据"),
    # Bigram
    ("00:22:52", "从最简单的 Bigram 神经网络开始", "模型目标", "03-Bigram语言模型"),
    ("00:23:26", "继承 nn.Module 并创建 embedding 表", "代码", "03-Bigram语言模型"),
    ("00:24:10", "Embedding 查表如何处理 B×T 索引", "代码与shape", "03-Bigram语言模型"),
    ("00:24:52", "把 65 维查表结果直接解释为 logits", "模型含义", "03-Bigram语言模型"),
    ("00:25:26", "Bigram 为什么只看当前 token", "能力限制", "03-Bigram语言模型"),
    ("00:26:02", "用 cross_entropy 衡量预测错误", "损失函数", "03-Bigram语言模型"),
    ("00:26:40", "cross_entropy 期望 channel 位于哪里", "API与shape", "03-Bigram语言模型"),
    ("00:27:15", "把 B×T×C 展平为 BT×C", "代码与shape", "03-Bigram语言模型"),
    ("00:27:54", "把 targets 展平并成功计算 loss", "代码与运行结果", "03-Bigram语言模型"),
    ("00:28:37", "用 ln(65) 判断初始 loss 是否合理", "数值解释", "03-Bigram语言模型"),
    ("00:29:14", "generate 循环如何逐 token 扩展序列", "生成流程", "03-Bigram语言模型"),
    ("00:29:56", "targets 可选时如何跳过 loss", "接口修改", "03-Bigram语言模型"),
    ("00:30:36", "为什么只取最后时间步 logits", "生成逻辑", "03-Bigram语言模型"),
    ("00:31:15", "softmax 把 logits 转为概率", "概率", "03-Bigram语言模型"),
    ("00:31:54", "multinomial 按概率采样下一个 token", "采样", "03-Bigram语言模型"),
    ("00:32:32", "cat 把新 token 拼回时间维", "代码与shape", "03-Bigram语言模型"),
    ("00:33:12", "用换行 token 启动第一次生成", "运行实验", "03-Bigram语言模型"),
    ("00:34:57", "未训练输出乱码与通用 generate 接口", "结果与接口设计", "03-Bigram语言模型"),
    ("00:35:34", "选择 AdamW 与合适学习率", "优化器", "04-训练循环与模型评估"),
    ("00:36:16", "标准训练循环的五个动作", "训练代码", "04-训练循环与模型评估"),
    ("00:36:58", "zero_grad backward step 分别做什么", "反向传播", "04-训练循环与模型评估"),
    ("00:38:08", "训练后 loss 降到约 2.5", "运行结果", "04-训练循环与模型评估"),
    ("00:39:02", "把 Notebook 整理成 bigram.py", "工程整理", "04-训练循环与模型评估"),
    ("00:40:34", "选择 device 并让数据模型位于同一设备", "设备与错误预防", "04-训练循环与模型评估"),
    ("00:41:20", "estimate_loss 为什么要平均多个 batch", "评估", "04-训练循环与模型评估"),
    ("00:42:24", "eval train 与 no_grad 各自影响什么", "评估模式", "04-训练循环与模型评估"),
    # 加权聚合数学铺垫
    ("00:43:02", "建立 B×T×C 玩具张量与因果通信目标", "问题定义", "05-Self-Attention"),
    ("00:43:50", "当前位置需要汇总自己和之前 token", "概念", "05-Self-Attention"),
    ("00:44:42", "为什么先用前缀平均作为通信基线", "数学动机", "05-Self-Attention"),
    ("00:45:28", "初始化 xbow 并遍历 batch 与时间", "朴素代码", "05-Self-Attention"),
    ("00:46:12", "切出 xprev 并沿时间维求平均", "代码与shape", "05-Self-Attention"),
    ("00:47:03", "对照 x 与 xbow 理解前缀平均结果", "运行结果", "05-Self-Attention"),
    ("00:47:48", "用 A@B 学习矩阵乘法的行列组合", "矩阵例子", "05-Self-Attention"),
    ("00:48:38", "全一行为什么会求出 B 的列和", "矩阵例子", "05-Self-Attention"),
    ("00:49:32", "下三角全一矩阵产生逐行前缀和", "数学推导", "05-Self-Attention"),
    ("00:50:28", "归一化下三角矩阵得到前缀平均", "数学推导", "05-Self-Attention"),
    ("00:51:22", "用具体数字验证加权前缀平均", "矩阵例子", "05-Self-Attention"),
    ("00:52:50", "构造 T×T 的归一化权重矩阵", "代码", "05-Self-Attention"),
    ("00:53:45", "T×T 与 B×T×C 的批量矩阵乘法", "代码与shape", "05-Self-Attention"),
    ("00:54:25", "allclose 验证循环版和矩阵版一致", "验证", "05-Self-Attention"),
    ("00:54:48", "批量矩阵乘法不会混合不同 batch", "shape与隔离", "05-Self-Attention"),
    ("00:55:34", "用零分数与 tril 改写第三版权重", "实现演进", "05-Self-Attention"),
    ("00:56:18", "masked_fill 用负无穷屏蔽未来", "因果掩码", "05-Self-Attention"),
    ("00:56:58", "softmax 还原归一化下三角权重", "运行结果", "05-Self-Attention"),
    ("00:57:45", "亲和度将从固定零分数变成数据依赖", "概念过渡", "05-Self-Attention"),
    ("00:58:05", "矩阵乘法完成按权重的信息聚合", "数学总结", "05-Self-Attention"),
    ("00:58:17", "从加权聚合过渡到 Self-Attention", "章节过渡", "05-Self-Attention"),
    # embedding 与 QK 引入，保持样章从 M086 开始
    ("00:59:21", "整理构造参数并用 n_embd 建立中间特征空间", "代码整理与模型重构", "05-Self-Attention"),
    ("01:00:00", "lm_head 把 embedding 映射回词表 logits", "代码与shape", "05-Self-Attention"),
    ("01:00:23", "区分 embedding 维 C 与 vocab 维 C", "shape与术语", "05-Self-Attention"),
    ("01:01:05", "创建 position embedding table", "位置编码", "05-Self-Attention"),
    ("01:01:26", "arange 生成位置索引并查表", "代码与shape", "05-Self-Attention"),
    ("01:02:02", "B×T×C 与 T×C 如何广播相加", "广播", "05-Self-Attention"),
    ("01:02:28", "Bigram 暂时用不上位置信息", "能力解释", "05-Self-Attention"),
    ("01:03:36", "均匀平均为何无法选择重要 token", "问题定义", "05-Self-Attention"),
    ("01:04:41", "用 query 与 key 产生数据依赖关系", "QK概念", "05-Self-Attention"),
    # 已冻结样章 M086-M093
    ("01:05:21", "把匹配程度变成可以学习的分数", "概念与数学", "05-Self-Attention"),
    ("01:06:03", "每个 token 独立生成 key 和 query", "代码与shape", "05-Self-Attention"),
    ("01:06:53", "为什么必须转置 key", "代码与shape", "05-Self-Attention"),
    ("01:07:44", "固定平均变成数据依赖权重", "概念与运行结果", "05-Self-Attention"),
    ("01:08:51", "用元音寻找辅音理解 QK", "概念类比", "05-Self-Attention"),
    ("01:09:24", "查看 softmax 之前的原始分数", "调试与运行结果", "05-Self-Attention"),
    ("01:10:04", "因果掩码与 softmax 各做一件事", "代码与模型行为", "05-Self-Attention"),
    ("01:11:08", "Value 才是最终被搬运的信息", "代码与shape", "05-Self-Attention"),
    # 注意力注解与缩放
    ("01:12:09", "注意力是有向图上的信息通信", "概念", "05-Self-Attention"),
    ("01:13:11", "自回归任务对应怎样的有向图", "图结构", "05-Self-Attention"),
    ("01:14:15", "注意力没有天然空间位置概念", "概念对比", "05-Self-Attention"),
    ("01:15:19", "batch 内有四组互不通信的图", "batch与shape", "05-Self-Attention"),
    ("01:16:23", "encoder 与 decoder 的 mask 差异", "架构对比", "05-Self-Attention"),
    ("01:17:18", "self-attention 与 cross-attention 的数据来源", "架构对比", "05-Self-Attention"),
    ("01:18:18", "为什么点积要除以根号 head_size", "Scaled Attention", "05-Self-Attention"),
    ("01:19:13", "方差过大会让 softmax 过度尖锐", "概率与初始化", "05-Self-Attention"),
    # 接入模型、多头与前馈
    ("01:20:17", "把单头注意力封装成 Head 并注册 tril buffer", "代码封装与PyTorch API", "06-Multi-Head-Attention"),
    ("01:20:51", "将 Head 接入 token 与位置表示之后", "模型接线", "06-Multi-Head-Attention"),
    ("01:21:37", "裁剪生成上下文并调整训练超参数", "生成逻辑与训练配置", "06-Multi-Head-Attention"),
    ("01:22:18", "单头注意力训练结果与局限", "运行结果", "06-Multi-Head-Attention"),
    ("01:23:03", "用 ModuleList 并行多个 Head 并按通道拼接", "概念与代码", "06-Multi-Head-Attention"),
    ("01:23:46", "四个八维头如何重新组成三十二维", "代码与shape", "06-Multi-Head-Attention"),
    ("01:24:45", "多头结果提升与 Transformer 图中组件对应", "运行结果与架构定位", "06-Multi-Head-Attention"),
    # Block 与残差
    ("01:25:28", "从论文图引出逐 token FeedForward", "架构与概念", "07-Transformer-Block"),
    ("01:26:12", "实现 Linear 加 ReLU 的逐 token 计算", "代码与shape", "07-Transformer-Block"),
    ("01:26:53", "FeedForward 将 loss 降至约 2.24 并引出 Block", "运行结果与架构", "07-Transformer-Block"),
    ("01:27:28", "Block 交替组织注意力通信与独立计算", "代码封装", "07-Transformer-Block"),
    ("01:27:55", "按 n_head 划分通道并顺序堆叠 Block", "代码与shape", "07-Transformer-Block"),
    ("01:28:34", "直接堆深后出现优化困难", "优化问题", "07-Transformer-Block"),
    ("01:29:14", "从 ResNet 图理解残差连接", "背景与概念", "07-Transformer-Block"),
    ("01:29:52", "加法让梯度沿残差主干直接传播", "梯度解释", "07-Transformer-Block"),
    ("01:30:30", "梯度高速公路为何帮助深层优化", "优化直觉", "07-Transformer-Block"),
    ("01:31:10", "实现 x 加 attention 和 x 加 feedforward", "残差代码", "07-Transformer-Block"),
    ("01:32:08", "输出投影与四倍宽 FeedForward", "代码与shape", "07-Transformer-Block"),
    # LayerNorm
    ("01:32:44", "残差与四倍 FFN 将 loss 降至约 2.08", "运行结果与过拟合", "07-Transformer-Block"),
    ("01:33:20", "从 BatchNorm 引出 LayerNorm", "归一化概念", "07-Transformer-Block"),
    ("01:33:58", "BatchNorm 为每个特征跨样本归一化", "数学与shape", "07-Transformer-Block"),
    ("01:34:36", "LayerNorm 改为对每个样本的特征归一化", "数学与shape", "07-Transformer-Block"),
    ("01:35:14", "LayerNorm 无需 running statistics 但保留 gamma beta", "实现差异与参数", "07-Transformer-Block"),
    ("01:35:52", "原论文 post-norm 与现代 pre-norm", "架构对比", "07-Transformer-Block"),
    ("01:36:23", "在 attention 与 FFN 前加入两个 LayerNorm", "代码接线", "07-Transformer-Block"),
    ("01:36:54", "LayerNorm 对每个 token 的 C 个特征统计", "shape", "07-Transformer-Block"),
    ("01:37:49", "LayerNorm 结果与末尾归一化完成 Transformer", "运行结果与模型完成", "07-Transformer-Block"),
    # Dropout 与规模化
    ("01:38:08", "用 n_layer 与 n_head 整理规模化参数", "代码整理", "08-完整GPT训练与生成"),
    ("01:38:30", "Dropout 可放在哪些残差分支末端", "正则化代码", "08-完整GPT训练与生成"),
    ("01:39:06", "随机失活为何近似训练子网络集成", "正则化概念", "08-完整GPT训练与生成"),
    ("01:39:34", "Dropout 通过随机子网络集成缓解过拟合", "正则化概念", "08-完整GPT训练与生成"),
    ("01:40:04", "放大 batch block_size 与 embedding 并降低学习率", "超参数与shape", "08-完整GPT训练与生成"),
    ("01:40:37", "六头六层与百分之二十 Dropout 配置", "超参数与运行过渡", "08-完整GPT训练与生成"),
    ("01:41:16", "A100 训练约十五分钟得到 1.48 并给出低配建议", "运行结果与硬件建议", "08-完整GPT训练与生成"),
    ("01:42:43", "生成文本像莎士比亚但无语义并结束编程部分", "结果分析与章节结束", "08-完整GPT训练与生成"),
    # encoder decoder
    ("01:43:20", "我们实现的是带三角 mask 的 decoder-only Transformer", "架构定位", "09-GPT与ChatGPT"),
    ("01:43:45", "原论文为何为机器翻译设计 encoder-decoder", "架构背景", "09-GPT与ChatGPT"),
    ("01:44:22", "翻译任务中的 START 与 END token", "序列设计", "09-GPT与ChatGPT"),
    ("01:44:50", "decoder 生成英文时还要条件化于法语输入", "decoder条件", "09-GPT与ChatGPT"),
    ("01:45:26", "encoder 无因果 mask 双向读取完整法语", "encoder", "09-GPT与ChatGPT"),
    ("01:45:53", "cross-attention 的 Q 来自 decoder 而 KV 来自 encoder", "cross-attention", "09-GPT与ChatGPT"),
    # nanoGPT
    ("01:46:26", "GPT 无外部条件只保留 decoder 并转入 nanoGPT", "架构总结", "09-GPT与ChatGPT"),
    ("01:46:56", "nanoGPT 的 train.py 包含哪些工程能力", "代码导览", "09-GPT与ChatGPT"),
    ("01:47:28", "nanoGPT CausalSelfAttention 与课堂 Head 的对应", "代码导览", "09-GPT与ChatGPT"),
    ("01:47:59", "四维张量把所有 head 作为并行维", "shape与优化", "09-GPT与ChatGPT"),
    ("01:48:30", "MLP GELU Block 与完整 GPT 如何对应课堂版本", "代码映射", "09-GPT与ChatGPT"),
    ("01:49:00", "checkpoint optimizer generate 工程细节与 ChatGPT 过渡", "章节过渡", "09-GPT与ChatGPT"),
    # GPT 到 ChatGPT
    ("01:49:32", "ChatGPT 训练先预训练再微调", "训练阶段总览", "09-GPT与ChatGPT"),
    ("01:50:03", "课堂模型约千万参数与字符数据的 token 换算", "规模与tokenizer", "09-GPT与ChatGPT"),
    ("01:50:34", "课堂约三十万 subword token 对比 GPT-3", "token规模", "09-GPT与ChatGPT"),
    ("01:51:06", "GPT-3 最大模型 175B 参数及结构超参数", "规模对比", "09-GPT与ChatGPT"),
    ("01:51:38", "GPT-3 训练约 300B token 并走向万亿规模", "数据规模", "09-GPT与ChatGPT"),
    ("01:52:08", "架构相近但大模型预训练需要数千 GPU", "基础设施", "09-GPT与ChatGPT"),
    ("01:52:38", "预训练只得到不稳定的互联网文档补全器", "能力边界", "09-GPT与ChatGPT"),
    ("01:53:11", "监督微调用问答示范塑造助手格式", "SFT", "09-GPT与ChatGPT"),
    ("01:53:42", "人工偏好排序训练奖励模型", "奖励模型", "09-GPT与ChatGPT"),
    ("01:54:14", "PPO 根据奖励模型优化生成策略", "强化学习", "09-GPT与ChatGPT"),
    ("01:54:46", "对齐把补全器变成助手且难以完整复现", "对齐总结", "09-GPT与ChatGPT"),
    # 总结
    ("01:55:19", "回顾 decoder-only Transformer 与约二百行代码资源", "课程与资源总结", "09-GPT与ChatGPT"),
    ("01:55:42", "GPT-3 架构相近但规模大数万至百万倍", "规模总结", "09-GPT与ChatGPT"),
    ("01:56:02", "任务型模型需要监督微调或更复杂对齐", "延伸方向", "09-GPT与ChatGPT"),
    ("01:56:15", "奖励模型与 PPO 仍是未展开的后续主题", "延伸总结", "09-GPT与ChatGPT"),
    ("01:56:20", "结束语与无新增知识画面", "结束与停顿", "09-GPT与ChatGPT"),
]


assert len(SEGMENTS) == 164, len(SEGMENTS)


def clock(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}.{int(seconds % 1 * 100):02d}"


timeline = []
start = "00:00:00"
for number, (end, title, content_type, chapter) in enumerate(SEGMENTS, 1):
    timeline.append({
        "micro_id": f"M{number:03d}",
        "start": start,
        "end": end,
        "title": title,
        "content_type": content_type,
        "chapter": chapter,
        "learning_goal": f"理解：{title}",
        "handling": "合并讲解" if content_type == "结束与停顿" else "正文讲解",
    })
    start = end

assert timeline[85]["micro_id"] == "M086"
assert timeline[85]["start"] == "01:04:41"
assert timeline[92]["end"] == "01:11:08"

with (ROOT / "qa/timeline.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=timeline[0].keys())
    writer.writeheader()
    writer.writerows(timeline)

caption_data = json.loads((ROOT / "sources/video.en-orig.json3").read_text())
caption_rows = []
index = 0
with (ROOT / "qa/semantic-review.csv").open(encoding="utf-8-sig") as file:
    review_status = {row["micro_id"]: row["status"] for row in csv.DictReader(file)}
for event in caption_data["events"]:
    text = "".join(part.get("utf8", "") for part in event.get("segs", [])).replace("\n", " ").strip()
    if not text:
        continue
    start_seconds = event.get("tStartMs", 0) / 1000
    while index + 1 < len(timeline) and start_seconds >= sec(timeline[index]["end"]):
        index += 1
    end_seconds = start_seconds + event.get("dDurationMs", 0) / 1000
    caption_rows.append({
        "subtitle_id": f"S{len(caption_rows) + 1:04d}",
        "start": clock(start_seconds),
        "end": clock(end_seconds),
        "micro_id": timeline[index]["micro_id"],
        "status": "已覆盖并人工复核"
        if review_status.get(timeline[index]["micro_id"]) == "已人工复核"
        else "已归属，待语义复核",
        "source_text": text,
    })

with (ROOT / "qa/subtitle-coverage.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=caption_rows[0].keys())
    writer.writeheader()
    writer.writerows(caption_rows)

print(f"已写入 {len(timeline)} 个微片段和 {len(caption_rows)} 个字幕事件")

# 原视频时间轴保留旧视频章节字段；随后同步生成独立的新学习章节映射和截图状态。
from build_learner_indexes import build_chapter_map, migrate_screenshot_index

build_chapter_map()
migrate_screenshot_index()
print("已同步学习章节映射和截图迁移字段")
