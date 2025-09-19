from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import os

# 基础检查：确保已启用 CUDA（否则 vLLM 在 CPU 上会非常慢）
if not torch.cuda.is_available():
    raise RuntimeError("未检测到 CUDA，请先在 WSL 中启用 GPU（nvidia-smi 正常、装好 CUDA/PyTorch）。")
print("CUDA 已启用")

model= "./model/Qwen/Qwen2.5-0.5B"

# 用与模型一致的分词器构造 chat prompt
tokenizer = AutoTokenizer.from_pretrained(model)

user_prompt = "你是什么模型？"

message = [
    {"role": "user", "content": user_prompt}
]

# 使用tokenize=False来获取字符串格式（而不是token ids）
prompts = tokenizer.apply_chat_template(
    message,
    tokenize=False,  # 返回字符串而不是token ids
    add_generation_prompt=True,  # 是否添加生成提示
    enable_thinking=True,   # 是否启用思考
)

print("prompts类型:", type(prompts))
print("prompts内容:", repr(prompts))

# 创建采样参数。
# temperature 控制生成文本的多样性，top_p 控制核心采样的概率，top_k 通过限制候选词的数量来控制生成文本的质量和多样性，min_p 通过设置概率阈值来筛选候选词，从而在保证文本质量的同时增加多样性
# max_tokens 用于限制模型在推理过程中生成的最大输出长度
# 对于思考模式，官方建议使用以下参数:temperature =0.6,TopP =0.95,TopK= 20，MinP = 0
# 对于非思考模式，官方建议使用以下参数:temperature=0.7,TopP =0.8,TopK= 20，MinP = 0

# 采样参数配置
stop_token_ids = [151645, 151643]  # Qwen2.5 的特殊停止token
stop_strings = ["<|im_end|>", "<|endoftext|>", "ContentLoaded"]  # 添加字符串停止词

sampling_params = SamplingParams(
    temperature=0.7,  # 提高温度增加多样性
    top_p=0.8,        # 降低top_p减少重复
    top_k=40,         # 增加top_k提供更多选择
    min_p=0.01,       # 设置最小概率阈值
    max_tokens=512,   # 减少最大长度避免过长生成
    stop_token_ids=stop_token_ids,
    stop=stop_strings, # 添加字符串停止条件
    repetition_penalty=1.1,  # 添加重复惩罚
)

# 初始化 vLLM 推理引擎（启用 GPU）
# - dtype="half": Ampere/RTX30 系列稳定，显存友好
# - tensor_parallel_size: 自动用到多 GPU（若只有一块则为1）
# - gpu_memory_utilization: 0.9 尽量吃满显存
llm = LLM(
    model=model,
    tokenizer=None, # 这里让 vLLM 自己管理内部 tokenizer；上面仅用于模板
    max_model_len=2048,  # 减少最大序列长度以节省显存
    trust_remote_code=True,
    dtype="half",
    tensor_parallel_size=1,  # 单GPU，明确设置为1
    gpu_memory_utilization=1,  # 提高到70%，为KV cache提供足够空间
    swap_space=2,  # 设置交换空间为2GB
    # enforce_eager=True,  # 先注释掉，让系统自动选择最优模式
)

outputs = llm.generate(
    prompts,
    sampling_params
)

# 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表
# 打印输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated Text: {generated_text}")
    print("-" * 50)