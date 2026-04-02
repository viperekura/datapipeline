# DataPipeline

语言模型训练数据集处理工具包。支持 PT / SFT / DPO 范式。

## 项目结构

```
pipeline/
├── tokenizer.py           # BPE tokenizer
├── text.py                # Text normalization
├── packing.py             # Sequence packing
├── io.py                  # File / HDF5 I/O
├── processors.py          # PT / SFT / DPO processors
├── export.py              # Dataset -> JSONL
├── cache.py               # JSONL -> Tokenize -> H5
├── utils.py               # Logging, error handling
└── strategies/            # Prompt strategy (strategy pattern)
    ├── base.py            #   PromptStrategy ABC
    ├── chatml.py          #   ChatML format (configurable tokens)
    ├── alpaca.py          #   Alpaca format (configurable tokens)
    └── factory.py         #   StrategyFactory
```

## 数据流

整体分为两个阶段，通过磁盘 JSONL 文件解耦：

```
Stage 1: Export Dataset                     Stage 2: Tokenize & Cache
┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
│                                      │    │                                      │
│  HuggingFace Dataset                 │    │  JSONL files (one dict per line)     │
│       │                              │    │       │                              │
│       ▼  process_func (optional)     │    │       ▼  json.loads()                │
│  export_dataset()                    │    │  Processor.process(input_dict)       │
│       │  chunk by chunk_size         │    │       │                              │
│       ▼                              │    │       │  ┌───────────────────────┐   │
│  ./dataset/prefix_chunk_0.jsonl ────────────────|>   │ PT / SFT / DPO        │   │
│  ./dataset/prefix_chunk_1.jsonl      │    │       │  │ processor details     │   │
│  ...                                 │    │       │  └───────────────────────┘   │
│                                      │    │       ▼                              │
│  modules: export.py                  │    │  List[Tensor]                        │
│                                      │    │       │                              │
│                                      │    │       ▼  SequencePacker (optional)   │
│                                      │    │  Fixed-length packed tensors         │
│                                      │    │       │                              │
│                                      │    │       ▼  IOHandler.save_h5()         │
│                                      │    │  ./cached/chunk_0.h5                 │
│                                      │    │  ./cached/chunk_1.h5                 │
│                                      │    │  ...                                 │
│                                      │    │                                      │
│                                      │    │  modules: io.py, processors.py,      │
│                                      │    │           packing.py, tokenizer.py   │
└──────────────────────────────────────┘    └──────────────────────────────────────┘
```

### Processor 转换规则

**PT (Pre-training)**
```
Input:  {"text": "Hello world"}
Action: tokenizer.encode(text + "<｜end▁of▁sentence｜>")
Output: {"sequence": Tensor[int32]}
```

**SFT (Supervised Fine-tuning)** — uses PromptStrategy
```
Input:  {"query": "...", "response": "..."}
Action:
  1. strategy.build_prompt(input_dict)  ->  "<｜im▁start｜>user\n...\n<｜im▁start｜>assistant\n"
  2. concat response + response_suffix
  3. tokenizer.encode full string
  4. build loss_mask: query part=False, response part=True
Output: {"sequence": Tensor[int32], "loss_mask": Tensor[bool]}
```

**DPO (Direct Preference Optimization)** — uses PromptStrategy
```
Input:  {"query": "...", "chosen": "...", "rejected": "..."}
Action:
  1. strategy.build_prompt(input_dict)  ->  shared query prompt
  2. encode chosen  = query + response_start + chosen  + response_suffix
  3. encode rejected = query + response_start + rejected + response_suffix
  4. build masks: query part=False, response part=True
Output: {"chosen": Tensor, "chosen_mask": Tensor[bool],
         "rejected": Tensor, "rejected_mask": Tensor[bool]}
```

### 序列打包

当 `pack_size > 0` 时启用，使用 First-Fit Decreasing 算法将变长序列填充到固定长度：

- 按序列长度降序排列
- 逐个放入当前包，超长则截断
- 当前包放不下时开新包
- 不足部分用 `pad_value` 填充

## 快速开始

### 1. 导出数据集为 JSONL

```python
from datasets import load_dataset
from pipeline import export_dataset

dataset = load_dataset("your-dataset")
export_dataset(
    dataset=dataset["train"],
    output_dir="./data",
    output_prefix="train",
    process_func=lambda x: {"text": x["content"]},
)
```

### 2. 分词并缓存为 HDF5

```python
from pipeline import BpeTokenizer, ProcessorFactory, cache_jsonl

tokenizer = BpeTokenizer("tokenizer.json")
processor = ProcessorFactory.create("pt", tokenizer)

cache_jsonl(
    files=["./data/train.jsonl"],
    output_dir="./cached",
    processor=processor,
    pack_size=4096,
)
```

### 3. 使用策略模式（默认 token）

```python
from pipeline import StrategyFactory, ProcessorFactory

strategy = StrategyFactory.create("alpaca")
processor = ProcessorFactory.create_with_strategy("sft", tokenizer, strategy)
```

### 4. 使用策略模式（自定义 token）

```python
# 自定义 ChatML 的特殊 token
strategy = StrategyFactory.create("chatml",
    user_start="<s>user\n",
    user_end="</s>\n",
    assistant_start="<s>assistant\n",
    assistant_end="</s>\n<｜end▁of▁sentence｜>",
)
processor = ProcessorFactory.create_with_strategy("sft", tokenizer, strategy)
```

## 策略格式

| Strategy  | Key        | Default Tokens                                                                                   |
|-----------|------------|--------------------------------------------------------------------------------------------------|
| ChatML    | `"chatml"` | `<｜im_start｜>user`, `<｜im_end｜>\n`, `<｜im_start｜>assistant`, `<｜im_end｜>\n`                |
| Alpaca    | `"alpaca"` | `### Instruction:`, `### Response:`, `<｜end▁of▁sentence｜>`                                    |

所有策略的 token 均可通过构造函数参数自定义，同时支持通过 `StrategyFactory.register()` 注册新格式。

## 命令行工具

```bash
# 缓存 JSONL 到 H5
python scripts/cache_h5.py pt ./dataset/chinese-c4-pretrain
python scripts/cache_h5.py sft ./dataset/belle-sft --pack-size 4096 --strategy alpaca
```

## 脚本示例

```
scripts/
├── cache_h5.py                              # Stage 2: JSONL -> H5 (CLI tool)
├── pre_train/
│   ├── chinese-c4.py                        # Chinese pretrain data export
│   ├── chinese-cosmopedia.py                # Chinese pretrain data export
│   ├── english-fineweb.py                   # English pretrain data export
│   └── english-wiki.py                      # English pretrain data export
├── supervised_finetuning/
│   ├── sft_belle.py                         # Belle Chinese SFT data export
│   ├── sft_chinese_instruct.py              # Chinese instruct SFT (with TextNormalizer)
│   ├── sft_coder.py                         # Code SFT data export
│   ├── sft_firefly-1.1m-rephrased.py        # Firefly SFT data export
│   └── sft_magpie-pro-300k.py               # Magpie Pro SFT data export
└── reforce_learning/
    └── dpp_chinese_dpo_pairs.py              # Chinese DPO preference pairs export
```