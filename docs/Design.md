# Pipeline 设计文档

## 概述

本项目是一个语言模型训练数据集处理工具包，将 HuggingFace 数据集转换为模型可直接消费的 HDF5 格式。整体分为两个阶段，通过磁盘 JSONL 文件解耦。

## 数据流架构

```
 Stage 1: Export                Stage 2: Tokenize & Cache
┌──────────────────────────┐    ┌──────────────────────────────────┐
│                          │    │                                  │
│  HuggingFace Dataset     │    │  JSONL files (one dict/line)     │
│       │                  │    │       │                          │
│       ▼  process_func    │    │       ▼  json.loads()            │
│  export_dataset()        │    │  Processor.process(input_dict)   │
│       │  chunk_size      │    │       │                          │
│       ▼                  │    │       ▼  SequencePacker (opt.)   │
│  prefix_chunk_0.jsonl ────────▶  IOHandler.save_h5()            │
│  prefix_chunk_1.jsonl    │    │       │                          │
│  ...                     │    │       ▼                          │
│                          │    │  chunk_0.h5                      │
│  module: io.py           │    │  chunk_1.h5                      │
│                          │    │                                  │
│                          │    │  modules: io.py,                 │
│                          │    │   processors/, packing.py,       │
│                          │    │   tokenizer.py                   │
└──────────────────────────┘    └──────────────────────────────────┘
```

### 阶段 1: export_dataset()

**职责**: 将 HuggingFace Dataset 导出为 JSONL 文件。

**流程**:
1. 按 `chunk_size`（默认 100 万条）将数据集分片
2. 每条样本通过可选的 `process_func` 转换（字段映射、文本清洗等）
3. `process_func` 返回单个 dict 或 list[dict]（支持一对多展开）
4. 每片写为一个 JSONL 文件

**模块**: `pipeline/io.py` (export_dataset 函数)

### 阶段 2: cache_jsonl()

**职责**: 将 JSONL 文件分词、打包、保存为 HDF5 格式。

**流程**:
1. 逐行读取 JSONL，解析为 dict
2. 通过 `Processor.process(input_dict)` 转换为 tensor dict
3. 可选通过 `SequencePacker` 打包为固定长度
4. 通过 `IOHandler.save_h5()` 写入 HDF5

**模块**: `pipeline/io.py`（编排）+ `pipeline/processors/`（转换）+ `pipeline/packing.py`（打包）+ `pipeline/tokenizer.py`（分词）

## 处理器设计

### ProcessorFactory

通过工厂模式创建不同类型的处理器：

| 方法 | 说明 |
|------|------|
| `create(processor_type, tokenizer)` | 创建处理器，SFT/DPO 使用默认 ChatML 策略 |
| `create_with_strategy(processor_type, tokenizer, strategy)` | 创建带自定义策略的处理器 |
| `create_with_strategy_name(processor_type, tokenizer, strategy_name, **kwargs)` | 通过名称创建策略，`**kwargs` 透传用于自定义 token |
| `register(processor_type)` | 注册自定义处理器（装饰器方式） |

### 处理器类型

**PreTrainProcessor** (`"pt"`)
```
Input:  {"text": "Hello world"}
Action: tokenizer.encode(text + "<｜end▁of▁sentence｜>")
Output: {"sequence": Tensor[int32]}
```

**SFTProcessor** (`"sft"`) — 注入 PromptStrategy
```
Input:  {"query": "...", "response": "..."}
Action:
  1. tokenizer.encode(query) + tokenizer.encode(response)
  2. strategy.assemble_prompt(query_tokens) -> prompt with format tokens
  3. strategy.assemble_response(response_tokens) -> response with suffix
  4. concat prompt + response, build loss_mask: query=False, response=True
Output: {"sequence": Tensor[int32], "loss_mask": Tensor[bool]}
```

**DPOProcessor** (`"dpo"`) — 注入 PromptStrategy
```
Input:  {"query": "...", "chosen": "...", "rejected": "..."}
Action:
  1. tokenizer.encode(query/chosen/rejected)
  2. strategy.assemble_prompt(query_tokens) -> shared prompt
  3. strategy.assemble_response(chosen_tokens) -> chosen with suffix
  4. strategy.assemble_response(rejected_tokens) -> rejected with suffix
  5. build masks: prompt=False, response=True
Output: {"chosen": Tensor, "chosen_mask": Tensor[bool],
         "rejected": Tensor, "rejected_mask": Tensor[bool]}
```

## 策略模式

### PromptStrategy

抽象 prompt/response 格式策略，将已分词的 token 列表与格式 token 组装。所有参数可通过构造函数自定义配置。

**接口**:
- `name` — 策略名称属性
- `assemble_prompt(query_tokens: List[int])` — 将 query tokens 组装为完整 prompt，包含格式 token 到 response 开始标记
- `assemble_response(response_tokens: List[int])` — 将 response tokens 包装格式 token（后缀、EOS 等）

**内置实现**:

ChatML:

| Parameter       | Default Token                              |
|-----------------|--------------------------------------------|
| user_start      | `<｜im_start｜>user\n`                     |
| user_end        | `<｜im_end｜>\n`                           |
| assistant_start | `<｜im_start｜>assistant\n`                |
| assistant_end   | `<｜im▁end｜>\n`                          |

Alpaca:

| Parameter        | Default Token             |
|------------------|---------------------------|
| instruction_start | `### Instruction:\n`     |
| response_start    | `### Response:\n`        |
| response_suffix   | `<｜end▁of▁sentence｜>` |

**自定义示例**:
```python
strategy = StrategyFactory.create("chatml", tokenizer,
    user_start="<s>user\n",
    user_end="</s>\n",
    assistant_start="<s>assistant\n",
    assistant_end="</s>\n",
)
```

**注册新格式**:
```python
from pipeline import StrategyFactory
from pipeline.strategies import PromptStrategy

class MyStrategy(PromptStrategy):
    @property
    def name(self) -> str:
        return "my_format"
    # ... 实现抽象方法

StrategyFactory.register("my_format", MyStrategy)
```

## 序列打包

`SequencePacker` 将变长序列打包为固定长度的 tensor，使用流式拼接算法（Streaming Concat）：

1. 验证并归一化所有序列（检查维度、统一 dtype）
2. 逐个将序列追加到缓冲区，当缓冲区长度 >= pack_size 时，截取完整块并保存
3. 循环结束后，将尾部不足 pack_size 的部分用 `pad_value` 填充

序列可能跨块拆分（这是 LLM 训练的常见做法，如 TRL、Megatron-LM）。

打包在 `cache_jsonl()` 中按 `pack_size` 启用（`> 0` 时生效，`<= 0` 跳过打包）。

## 辅助模块

### TextNormalizer (`pipeline/text.py`)

文本标准化：统一标点符号（弯引号→直引号、破折号、省略号等），通过 `DEFAULT_REPLACEMENTS` 配置，支持 `custom_rules` 扩展。

### IOHandler (`pipeline/io.py`)

- `save_h5(output_dir, file_name, tensor_group)` — 按 key 分组存储 tensor 到 HDF5
- `load_h5(file_path, share_memory)` — 加载 HDF5，支持共享内存（用于 DataLoader 多进程）
- `fetch_files(directory)` / `fetch_folders(root_dir)` — 文件/目录遍历

### AutoTokenizer (`pipeline/tokenize/tokenizer.py`)

基于 HuggingFace `tokenizers` 库的 BPE 分词器，支持从文件加载、训练、保存。内置 `<｜begin▁of▁sentence｜>`/`<｜end▁of▁sentence｜>`/`<｜▁pad▁｜>` 控制符和 `<｜im▁start｜>`/`<｜im▁end｜>` 特殊 token。

支持动态属性访问（`bos_token`, `bos_token_id`, `stop_ids`, `pad_id` 等）和聊天模板（`set_chat_template`, `apply_chat_template`）。

## API 参考

各模块的详细 API 文档请参阅对应的源文件。
