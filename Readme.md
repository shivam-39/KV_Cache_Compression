# Structured KV Cache Compression with Adaptive Ranking for Efficient LLM Inference

A method for efficient KV cache compression in Large Language Models combining **structured token partitioning** with **layer-wise adaptive low-rank decomposition**. This approach achieves **5.6% relative accuracy improvement** over prior GEAR-based compression while using only slightly more memory.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/username/GEAR.git
cd GEAR

# Create virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build CUDA kernels (optional, for quantization)
cd cuda_supported_gear/quant
python setup.py install
cd ../..
```

### 3. Run a Quick Test

```bash
cd GenerationBench/GenerationTest

# Run GSM8K evaluation with our adaptive ranking method
python evaluation_gsm8k_true_compression.py --model meta-llama/Meta-Llama-3-8B --compression 4-bit

# Expected: ~47.95% accuracy
```

## 💻 Running Experiments

### Basic Usage: Single Benchmark

```bash
cd GenerationBench/GenerationTest

# Evaluate GSM8K with adaptive ranking at 4-bit compression
python evaluation_gsm8k_true_compression.py \
    --model meta-llama/Meta-Llama-3-8B \
    --compression 4-bit \
    --cache_dir ~/.cache/huggingface \
    --output_dir ./results/

# Evaluate AQuA with adaptive ranking
python evaluation_aqua_cot_true_compression.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --compression 2-bit
```

### Run All Experiments

```bash
# Replicate all main results (takes ~2-4 hours on single GPU)
bash run_xyp_and_adaptiverank.sh

# Individual model runs
bash run_template_llama-3-8b.sh      # LLaMA-3-8B experiments
bash run_mistral-7b.sh               # Mistral-7B experiments
```

### Evaluate Specific Configuration

```python
# Python API for custom evaluation
from GenerationBench.GenerationTest.GEARLM.TrueCompression import GEARCompression
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Initialize compression with adaptive ranking
compressor = GEARCompression(
    model=model,
    compression_bits=4,
    adaptive_ranking=True,  # Key: Enable adaptive ranking
    tau=0.90,              # Energy preservation threshold
    prefill_size=4,        # Prefill region tokens
    recency_size=32,       # Recency region tokens
    buffer_size=16         # Buffering for amortization
)

# Run inference
output = compressor.generate(inputs, max_length=128)
```

## Acknowledgments

This work was supported by UC San Diego. We thank Meta for providing LLaMA models and Mistral AI for the Mistral-7B model. Experiments were conducted on UC San Diego's computing resources.

---

**Last Updated**: May 2026  
**Status**: Active Development
