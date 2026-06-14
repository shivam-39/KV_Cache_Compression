# Structured KV Cache Compression with Adaptive Ranking for Efficient LLM Inference

A novel method for efficient KV cache compression in Large Language Models combining **structured token partitioning** with **layer-wise adaptive low-rank decomposition**. This approach achieves **5.6% relative accuracy improvement** over prior GEAR-based compression while using only slightly more memory.

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- 24GB+ GPU memory (for model + KV cache)

Key dependencies:
```
accelerate>=0.27.2
torch>=2.0.0
transformers>=4.35.0
datasets>=2.18.0
numpy>=1.24.0
pandas>=2.0.0
```

See `requirements.txt` for complete list.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/username/GEAR.git
cd GEAR

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build CUDA kernels (optional, for quantization)
cd cuda_supported_gear/quant
python setup.py install
cd ../..
```

### 2. Download Models

```bash
# Models will be auto-downloaded via Hugging Face Transformers
# First run will cache them in ~/.cache/huggingface/hub/

# Supported models:
# - meta-llama/Llama-2-7b-hf
# - meta-llama/Meta-Llama-3-8B
# - mistralai/Mistral-7B-Instruct-v0.3
```

### 3. Run a Quick Test

```bash
cd GenerationBench/GenerationTest

# Run GSM8K evaluation with our adaptive ranking method
python evaluation_gsm8k_true_compression.py --model meta-llama/Meta-Llama-3-8B --compression 4-bit

# Expected: ~47.95% accuracy, completes in ~5-10 minutes
```

## 📁 Project Structure

```
GEAR/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                             # License
│
├── cuda_supported_gear/                # Core quantization & compression code
│   ├── modeling_llama_kivi.py         # LLaMA with KIVI quantization
│   ├── modeling_llamagear.py          # LLaMA with GEAR framework
│   └── quant/                         # Quantization operators
│       ├── qmodule.py                 # Core quantization module
│       ├── gemv.py                    # GPU kernels for quantized inference
│       ├── matmul.py                  # Quantized matrix multiplication
│       ├── new_pack.py                # Packing utilities + adaptive ranking
│       └── csrc/                      # CUDA source (gemv kernels)
│
├── GenerationBench/                    # Experimental evaluation suite
│   └── GenerationTest/                # Main benchmark directory
│       ├── GEARLM/                    # Compression strategies
│       │   ├── Simulated/             # Simulated compression
│       │   └── TrueCompression/       # True (actual) compression
│       ├── evaluation_gsm8k.py        # GSM8K benchmark (5-shot CoT)
│       ├── evaluation_aqua_cot.py     # AQuA benchmark (8-shot)
│       ├── evaluation_bbh_cot.py      # BBH benchmark (3-shot)
│       ├── lib_prompt/                # Benchmark prompts
│       ├── outputs/                   # Results directory
│       └── run_*.sh                   # Experiment scripts
│
├── research_paper.md                   # Full 6-page research paper (Markdown)
├── research_paper_latex_template.tex   # Conference-ready LaTeX (14 pages)
├── research_paper_latex_template.pdf   # Generated PDF
│
└── Documentation/
    ├── PAPER_SUMMARY.md               # Technical overview
    ├── LATEX_EXPANSION_SUMMARY.md     # LaTeX expansion details
    ├── CONVERSION_GUIDE.md            # PDF/DOCX generation
    ├── COMPLETION_SUMMARY.md          # Project completion checklist
    └── MANIFEST.md                    # File index
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

## 🔍 Replicating Specific Results

### Replicate 4-bit Compression Results

```bash
cd GenerationBench/GenerationTest

# LLaMA-3-8B with adaptive ranking
python evaluation_gsm8k_true_compression.py --model meta-llama/Meta-Llama-3-8B --compression 4-bit

# Mistral-7B with adaptive ranking
python evaluation_gsm8k_true_compression.py --model mistralai/Mistral-7B-Instruct-v0.3 --compression 4-bit
```

### Replicate 2-bit Compression Results

```bash
# LLaMA-3-8B at 2-bit
python evaluation_gsm8k_true_compression.py --model meta-llama/Meta-Llama-3-8B --compression 2-bit

# Mistral-7B at 2-bit
python evaluation_gsm8k_true_compression.py --model mistralai/Mistral-7B-Instruct-v0.3 --compression 2-bit
```

<!-- ### Ablation Study Replication

Each ablation can be run by modifying hyperparameters:

```bash
# No adaptive ranking (use fixed rank across all layers)
python evaluation_gsm8k_true_compression.py --model meta-llama/Meta-Llama-3-8B --compression 4-bit --fixed-rank 64

# No token partitioning (compress all tokens equally)
python evaluation_gsm8k_true_compression.py --model meta-llama/Meta-Llama-3-8B --compression 4-bit --prefill-size 0 --recency-size 0

# No buffering (per-token compression)
python evaluation_gsm8k_true_compression.py --model meta-llama/Meta-Llama-3-8B --compression 4-bit --buffer-size 1
``` -->


## 🐛 Troubleshooting

### Out of Memory (OOM) Error

```python
# Reduce batch size
python evaluation_gsm8k_true_compression.py --batch_size 1

# Use smaller model
python evaluation_gsm8k_true_compression.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Enable gradient checkpointing
python evaluation_gsm8k_true_compression.py --gradient_checkpointing
```

### Different Results

Results may vary slightly due to:
- Different random seeds (set `--seed 42` for reproducibility)
- Different prompt formats (we use standard 5/8/3-shot formats)
- Hardware differences (different GPU = slightly different quantization rounding)


## Acknowledgments

This work was supported by UC San Diego. We thank Meta for providing LLaMA models and Mistral AI for the Mistral-7B model. Experiments were conducted on UC San Diego's computing resources.

---

**Last Updated**: June 2024  
**Status**: Active Development
