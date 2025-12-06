"""Model loading and inference utilities with MPS optimization."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from typing import Optional
import os
from .config import load_config, get_device


class EmailModel:
    """Wrapper for LoRA-tuned email generation model."""
    
    def __init__(self, model_path: Optional[str] = None, use_quantization: bool = True):
        self.config = load_config()
        self.device = get_device()
        self.use_quantization = use_quantization and self.config['inference']['use_quantization']
        
        # Load base model
        base_model_name = self.config['model']['base_model']
        
        # Check for Hugging Face authentication if needed
        # Support environment variable, Streamlit secrets (for Streamlit Cloud), and CLI login (for local)
        hf_token = None
        
        # Try Streamlit secrets first (for Streamlit Cloud)
        try:
            import streamlit as st
            # Check if we're in a Streamlit context and secrets are available
            if hasattr(st, 'secrets'):
                try:
                    # Try multiple access methods for Streamlit secrets
                    # Method 1: Dictionary access with get()
                    hf_token = st.secrets.get("HF_TOKEN") or st.secrets.get("HUGGINGFACE_TOKEN")
                    # Method 2: Direct dictionary access
                    if not hf_token:
                        try:
                            hf_token = st.secrets["HF_TOKEN"]
                        except (KeyError, TypeError):
                            pass
                    # Method 3: Attribute access
                    if not hf_token:
                        try:
                            hf_token = getattr(st.secrets, "HF_TOKEN", None) or getattr(st.secrets, "HUGGINGFACE_TOKEN", None)
                        except (AttributeError, TypeError):
                            pass
                except (AttributeError, KeyError, TypeError, RuntimeError) as e:
                    # Secrets might not be available at initialization time
                    pass
        except (ImportError, RuntimeError, AttributeError):
            # Not in Streamlit environment or Streamlit not initialized yet
            pass
        
        # Fall back to environment variables (for Hugging Face Spaces, local, etc.)
        if not hf_token:
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_API_TOKEN")
        
        if "llama" in base_model_name.lower():
            try:
                from huggingface_hub import whoami, login
                # If token is in environment, use it
                if hf_token:
                    login(token=hf_token, add_to_git_credential=False)
                whoami()  # Test authentication
            except Exception as e:
                if not hf_token:
                    print(f"⚠️  Warning: Hugging Face authentication may be required for {base_model_name}")
                    print("   For local: Run: huggingface-cli login")
                    print("   For cloud: Set HF_TOKEN environment variable in Streamlit Cloud")
                    print(f"   Error: {e}")
        
        try:
            # Use token if available
            tokenizer_kwargs = {}
            if hf_token:
                tokenizer_kwargs["token"] = hf_token
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, **tokenizer_kwargs)
        except Exception as e:
            if "authentication" in str(e).lower() or "401" in str(e):
                error_msg = f"Authentication required for {base_model_name}.\n"
                error_msg += "For local: Run: huggingface-cli login\n"
                error_msg += "For Streamlit Cloud: Set HF_TOKEN as a secret in your app settings"
                raise RuntimeError(error_msg) from e
            raise
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # For Llama models, ensure tokenizer has padding side set
        self.is_llama = "llama" in base_model_name.lower()
        if self.is_llama:
            self.tokenizer.padding_side = "left"  # Left padding for causal models
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
        
        # Load model - quantization doesn't work on MPS, so disable it
        # For MPS, use float16 which is faster than float32
        # Use token if available
        model_kwargs = {}
        if hf_token:
            model_kwargs["token"] = hf_token
        
        # Check if CUDA is available for quantization (quantization requires GPU)
        cuda_available = torch.cuda.is_available()
        
        if self.use_quantization and cuda_available:
            # 8-bit quantization for GPU only (CUDA)
            # Note: Quantization doesn't work well on CPU, so we disable it for CPU-only environments
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    **model_kwargs
                )
            except Exception as e:
                # If quantization fails, fall back to standard loading
                print(f"⚠️  Quantization failed: {e}. Falling back to standard loading.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    **model_kwargs
                )
                if self.device != "cpu":
                    self.model.to(self.device)
        else:
            # Standard loading for MPS/CUDA - use float16 for speed
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    low_cpu_mem_usage=True,  # Faster loading
                    **model_kwargs
                )
                if self.device != "cpu":
                    self.model.to(self.device)
            except Exception as e:
                if "authentication" in str(e).lower() or "401" in str(e):
                    error_msg = f"Authentication required for {base_model_name}.\n"
                    error_msg += "For local: Run: huggingface-cli login\n"
                    error_msg += "For Streamlit Cloud: Set HF_TOKEN as a secret in your app settings"
                    raise RuntimeError(error_msg) from e
                raise
        
        # Load LoRA weights if available
        if model_path and os.path.exists(model_path):
            self.model = PeftModel.from_pretrained(self.model, model_path)
            print(f"Loaded LoRA weights from {model_path}")
        elif model_path:
            print(f"Warning: LoRA path {model_path} not found, using base model")
        
        # Set to evaluation mode and disable gradients for speed
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Compile model for MPS if available (PyTorch 2.0+)
        # This can provide 20-30% speedup on MPS
        # Note: torch.compile may not work on all MPS setups, so we catch errors
        try:
            if self.device == "mps" and hasattr(torch, "compile"):
                # Check PyTorch version (compile requires 2.0+)
                if hasattr(torch, "__version__"):
                    version_parts = torch.__version__.split(".")
                    major_version = int(version_parts[0])
                    if major_version >= 2:
                        print("Compiling model for MPS acceleration...")
                        try:
                            self.model = torch.compile(self.model, mode="reduce-overhead")
                            print("✅ Model compiled successfully")
                        except RuntimeError as compile_error:
                            # MPS compile might not be supported, fall back to uncompiled
                            print(f"⚠️  Model compilation not supported on this MPS setup, using uncompiled model")
                            print(f"   (This is normal and won't affect functionality)")
        except Exception as e:
            # Silently continue - compilation is optional
            pass
        
        self._base_model = None  # Store base model for comparison
    
    def _get_base_model(self):
        """Get the base model without LoRA weights for comparison."""
        if self._base_model is None:
            # Extract base model from PeftModel if it's wrapped
            if isinstance(self.model, PeftModel):
                self._base_model = self.model.get_base_model()
            else:
                self._base_model = self.model
        return self._base_model
    
    def generate_email_base(self, recipient: str, purpose: str, key_points: str, tone: str = "professional") -> str:
        """Generate email using base model (no LoRA fine-tuning) for comparison."""
        # Use base model instead of fine-tuned
        original_model = self.model
        self.model = self._get_base_model()
        
        try:
            result = self.generate_email(recipient, purpose, key_points, tone)
        finally:
            # Restore original model
            self.model = original_model
        
        return result
    
    def _format_prompt(self, recipient: str, purpose: str, key_points: str, tone: str) -> str:
        """Format prompt based on model type. key_points parameter kept for API compatibility but not used."""
        if self.is_llama:
            # Simplified format - no recipient or key_points section
            instruction = f"""Write a {tone} email.

Message: {purpose}

Please write a complete, professional email that includes:
- Appropriate greeting
- Clear body paragraphs addressing the message
- Professional closing

Email:"""
            
            # Use chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": "You are a professional email writing assistant. Write clear, concise, and professional emails."},
                    {"role": "user", "content": instruction}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt = instruction
        else:
            # Simple format for GPT-2 and other models - no recipient or key_points section
            prompt = f"""Write a {tone} email.

Message: {purpose}

Email:
"""
        
        return prompt
    
    def generate_email(
        self,
        recipient: str,
        purpose: str,
        key_points: str,
        tone: str = "professional"
    ) -> str:
        """Generate an email based on the provided context."""
        # Format prompt based on model type
        prompt = self._format_prompt(recipient, purpose, key_points, tone)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['model']['max_length']
        )
        
        # Move to device
        if self.device != "cpu" or not self.use_quantization:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation parameters - optimized for maximum speed
        # Use greedy decoding for fastest generation (can switch to sampling if quality suffers)
        use_greedy = self.config.get('inference', {}).get('use_greedy_decoding', False)
        
        generation_kwargs = {
            "max_new_tokens": 120 if self.is_llama else 80,  # Increased to ensure complete emails
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,  # Critical for speed - enables KV cache
        }
        
        if use_greedy:
            # Greedy decoding - fastest option
            generation_kwargs.update({
                "do_sample": False,
                "num_beams": 1,
            })
        else:
            # Sampling - balanced speed/quality
            generation_kwargs.update({
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,  # Reduced from 50 for speed
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 2,
            })
        
        # Llama-specific optimizations
        if self.is_llama:
            # Ensure use_cache is enabled (critical for speed)
            generation_kwargs["use_cache"] = True
        
        # Generate with timeout protection
        import time
        start_time = time.time()
        
        # Print progress for debugging
        decoding_mode = "greedy" if not generation_kwargs.get("do_sample", True) else "sampling"
        print(f"Starting generation (mode: {decoding_mode}, max_new_tokens={generation_kwargs['max_new_tokens']})")
        
        # MPS-specific optimizations
        if self.device == "mps":
            # Ensure MPS operations are optimized
            torch.mps.empty_cache()  # Clear cache before generation
        
        with torch.no_grad():
            try:
                # Use torch.inference_mode() for additional speed if available (PyTorch 1.9+)
                # Fall back to no_grad if not available
                if hasattr(torch, "inference_mode"):
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **inputs,
                            **generation_kwargs
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                gen_time = time.time() - start_time
                print(f"Generation completed in {gen_time:.1f} seconds")
                if gen_time > 30:  # Warn if generation takes too long
                    print(f"⚠️  Warning: Generation took {gen_time:.1f} seconds (expected: 5-10s with optimizations)")
            except Exception as e:
                gen_time = time.time() - start_time
                print(f"❌ Generation failed after {gen_time:.1f} seconds: {e}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                # Clear MPS cache after generation
                if self.device == "mps":
                    torch.mps.empty_cache()
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the email part
        if self.is_llama:
            # For Llama with chat template, extract after the assistant response
            if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
                email = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            elif "Email:" in generated_text:
                email = generated_text.split("Email:")[-1].strip()
            else:
                # Try to extract after the prompt
                email = generated_text[len(prompt):].strip()
        else:
            # For GPT-2 and others - extract after "Email:"
            if "Email:" in generated_text:
                email = generated_text.split("Email:")[-1].strip()
            else:
                # Fallback: extract after prompt
                email = generated_text[len(prompt):].strip()
        
        # Clean up any remaining special tokens or artifacts
        email = email.split("<|eot_id|>")[0].strip()  # Remove Llama EOT tokens
        email = email.split("</s>")[0].strip()  # Remove EOS tokens
        
        # Remove common artifacts and noise
        email = email.replace("Contact: contact@example.com", "")
        email = email.replace("email@example.com", "")
        
        # Remove phone number patterns (common GPT-2 hallucination)
        import re
        email = re.sub(r'\+\d+[\s-]?\d+[\s-]?\d+[\s-]?\d+[\s-]?\d+', '', email)
        email = re.sub(r'\d{3}[\s-]?\d{3}[\s-]?\d{4}', '', email)
        
        # Stop at common email endings - but preserve the full ending
        endings = [
            "\n\nBest regards", "\n\nRegards", "\n\nSincerely", 
            "\n\nThank you", "\n\nYours sincerely", "\n\nKind regards",
            "\n\nWarm regards", "\n\nBest", "\n\nThanks"
        ]
        found_ending = False
        for ending in endings:
            if ending.lower() in email.lower():
                # Find the first occurrence and keep everything up to and including the ending
                idx = email.lower().find(ending.lower())
                # Keep the ending and a bit after (for name/signature)
                email = email[:idx + len(ending)]
                found_ending = True
                break
        
        # If email ends abruptly (no proper ending), try to complete it
        if not found_ending:
            # Check if email ends mid-sentence (no punctuation at end)
            email_stripped = email.strip()
            if email_stripped and email_stripped[-1] not in ['.', '!', '?']:
                # Try to find the last complete sentence
                sentences = re.split(r'([.!?])\s+', email)
                if len(sentences) > 1:
                    # Reconstruct with proper punctuation
                    complete_sentences = []
                    for i in range(0, len(sentences) - 1, 2):
                        if i + 1 < len(sentences):
                            complete_sentences.append(sentences[i] + sentences[i + 1])
                    if complete_sentences:
                        email = ' '.join(complete_sentences)
                        # Add a proper closing if missing
                        if not any(ending.lower() in email.lower() for ending in endings):
                            email += "\n\nBest regards"
                else:
                    # No sentences found, just add period if missing
                    if email_stripped and not email_stripped.endswith('.'):
                        email = email.rstrip() + '.'
        
        # If email is too short or looks like gibberish, try to extract meaningful parts
        if len(email) < 20 or not any(char.isalpha() for char in email[:50]):
            # Try to find first sentence
            sentences = re.split(r'[.!?]\s+', email)
            if sentences:
                email = '. '.join(sentences[:3]) + '.'
        
        # Remove excessive newlines and clean up
        lines = []
        for line in email.split("\n"):
            line = line.strip()
            if line or (lines and lines[-1]):  # Keep single blank lines
                lines.append(line)
        email = "\n".join(lines)
        
        return email.strip()

