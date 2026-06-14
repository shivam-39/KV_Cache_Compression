from .TrueCompression import GearLlamaForCausalLM
from .TrueCompression import GearLlamaForCausalLMNew
from .TrueCompression import GearMistralForCausalLMNew

from .Simulated import CompressionConfig
from .Simulated import SimulatedGearLlamaForCausalLM

# from .modeling_llama_h2o import H2OLlamaForCausalLM, LlamaConfig
from .Simulated import SimulatedGearMistralForCausalLM, MistralConfig
try:
    from .Simulated import LlamaForCausalLMH2O
except ImportError:
    LlamaForCausalLMH2O = None
