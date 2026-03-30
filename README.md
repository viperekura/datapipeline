# DataPipeline

数据集处理工具，支持预训练 / SFT / DPO 三种训练范式。

## 项目结构

```
pipeline/
├── tokenizer.py      # BPE 分词器
├── text.py           # 文本规范化
├── packing.py        # 序列打包
├── io.py             # 文件/HDF5 读写
├── processors.py     # PT / SFT / DPO 处理器
├── export.py         # Dataset → JSONL
└── cache.py          # JSONL → Tokenize → H5
```

## 设计理念

模块**独立可用**，通过磁盘文件解耦，按需组合：

```
Dataset → export_dataset() → JSONL → cache_jsonl() → HDF5
                              ↑
                        processors.py
                        packing.py
                        io.py
```

## 使用方法

### 1. 导出数据集

```python
from datasets import load_dataset
from pipeline.export import export_dataset

dataset = load_dataset("your-dataset")
export_dataset(
    dataset=dataset["train"],
    output_dir="./data",
    output_prefix="train",
    process_func=lambda x: {"text": x["content"]},  # 可选
)
```

### 2. Tokenize 并缓存

```python
from pipeline import BpeTokenizer, ProcessorFactory, cache_jsonl

tokenizer = BpeTokenizer("tokenizer.json")
processor = ProcessorFactory.create("pt", tokenizer)  # "pt" | "sft" | "dpo"

cache_jsonl(
    files=["./data/train.jsonl"],
    output_dir="./cached",
    processor=processor,
    pack_size=4096,  # <=0 不打包
    pad_value=1,
)
```

### 3. 处理器类型

| 类型 | key | 输入 | 输出 |
|------|-----|------|------|
| 预训练 | `"pt"` | `{"text": "..."}` | `["sequence"]` |
| SFT | `"sft"` | `{"query": "...", "response": "..."}` | `["sequence", "loss_mask"]` |
| DPO | `"dpo"` | `{"query": "...", "chosen": "...", "rejected": "..."}` | `["chosen", "chosen_mask", "rejected", "rejected_mask"]` |

## 参数参考

### export_dataset()

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `dataset` | Dataset | 必填 | HuggingFace Dataset |
| `output_dir` | str | 必填 | 输出目录 |
| `output_prefix` | str | 必填 | 文件名前缀 |
| `chunk_size` | int | 1_000_000 | 每个文件的最大样本数 |
| `max_chunks` | int | None | 最大 chunk 数量 |
| `process_func` | callable | None | 样本转换函数 |
| `column` | str | "text" | 默认文本列名 |

### cache_jsonl()

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `files` | List[str] | 必填 | JSONL 文件列表 |
| `output_dir` | str | 必填 | 输出目录 |
| `processor` | BaseProcessor | 必填 | 处理器实例 |
| `pack_size` | int | -1 | 打包长度，<=0 不打包 |
| `pad_value` | int | 1 | 填充值 |

### SequencePacker

```python
packer = SequencePacker(pack_size=4096, pad_value=0)
packed = packer.pack([tensor1, tensor2, ...])  # → List[Tensor]
```

### IOHandler

```python
# 保存
IOHandler.save_h5("./out", "name", {"key": [tensor1, tensor2]})

# 加载
data = IOHandler.load_h5("./out")  # → {"key": [tensor1, ...]}
```
