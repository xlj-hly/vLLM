from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os
import time

load_dotenv()

token = os.getenv("HUGGINGFACE_HUB_TOKEN")

print(f"HuggingFace Token 状态: {'已设置' if token else '未设置'}")
print("-" * 50)

start_time = time.time()
print(f"开始下载")

# BAAI/bge-m3 是 HuggingFace Models 上的模型仓库，不是 dataset。
# 因此 repo_type 应为 "model"（或省略，默认也是 model）。
# 参数说明：
# - repo_id: 仓库ID，这里是 "BAAI/bge-m3"
# - repo_type: 仓库类型，模型应使用 "model"
# - cache_dir: 本地缓存目录
# - local_dir: 本地落地目录（扁平化存放文件，便于本地直接加载）
# - local_dir_use_symlinks: 是否用符号链接，False 兼容性更好
# - resume_download: 支持断点续传
# 访问私有或限流资源时，请通过环境变量 HUGGINGFACE_HUB_TOKEN 登录，不要把 token 写进代码。

# snapshot_download(
#     repo_id="BAAI/bge-m3",                # 仓库ID，bge-m3 向量模型
#     repo_type="model",                    # 模型仓库
#     local_dir="./model/BAAI/bge-m3",            # 本地落地目录（非 cache 结构）
#     cache_dir="./cache/BAAI/bge-m3",
#     local_dir_use_symlinks=False,           # 不用符号链接，直接复制
#     resume_download=True,                    # 支持断点续传
#     token=token  
# )

# snapshot_download(
#     repo_id="Qwen/Qwen2.5-3B",                # 仓库ID，bge-m3 向量模型
#     repo_type="model",                    # 模型仓库
#     local_dir="./model/Qwen/Qwen2.5-3B",            # 本地落地目录（非 cache 结构）
#     cache_dir="./cache/Qwen/Qwen2.5-3B",
#     local_dir_use_symlinks=False,           # 不用符号链接，直接复制
#     resume_download=True,                    # 支持断点续传
#     token=token  
# )

# snapshot_download(
#     repo_id="Systran/faster-whisper-medium",                # 仓库ID，bge-m3 向量模型
#     repo_type="model",                    # 模型仓库
#     local_dir="./model/faster-whisper/faster-whisper-medium",            # 本地落地目录（非 cache 结构）
#     cache_dir="./cache/faster-whisper/faster-whisper-medium",
#     local_dir_use_symlinks=False,           # 不用符号链接，直接复制
#     resume_download=True,                    # 支持断点续传
#     token=token 
# )

snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B",    # 模型仓库ID
    repo_type="model",    # 模型仓库类型
    local_dir="./model/Qwen/Qwen2.5-1.5B",    # 本地落地目录
    cache_dir="./cache/Qwen/Qwen2.5-1.5B",    # 本地缓存目录
    local_dir_use_symlinks=False,    # 不用符号链接，直接复制
    resume_download=True,    # 支持断点续传
    token=token 
)

end_time = time.time()
print(f"下载完成！耗时: {end_time - start_time:.2f} 秒")
