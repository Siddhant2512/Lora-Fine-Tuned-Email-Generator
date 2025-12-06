"""Streamlit UI for LoRA-Mail Assistant."""
import streamlit as st
import requests
from pathlib import Path
import sys

# Add parent to path for config
sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config
from src.model_loader import EmailModel

# Page config
st.set_page_config(
    page_title="LoRA-Mail Assistant",
    page_icon="✉️",
    layout="centered"
)

# Load config
config = load_config()
api_url = f"http://localhost:{config['api']['port']}"

def get_model():
    """Load fine-tuned model once and cache it in session state."""
    if st.session_state.email_model is None or not st.session_state.model_loaded:
        import time
        start_time = time.time()
        model_path = config['training']['output_dir']
        
        status = st.empty()
        status.info("🔄 Loading model (first time only, ~20 seconds)...")
        
        try:
            st.session_state.email_model = EmailModel(
                model_path=model_path if Path(model_path).exists() else None,
                use_quantization=config['inference']['use_quantization']
            )
            load_time = time.time() - start_time
            st.session_state.model_loaded = True
            status.success(f"✅ Model loaded in {load_time:.1f} seconds")
            time.sleep(0.5)  # Brief pause to show message
            status.empty()
        except Exception as e:
            error_msg = str(e)
            status.error(f"❌ Failed to load model: {error_msg}")
            
            # Provide helpful error messages
            if "authentication" in error_msg.lower() or "401" in error_msg.lower():
                st.error("**Authentication Error**")
                st.info("""
                **To fix this:**
                1. Go to Streamlit Cloud → Settings → Secrets
                2. Add your Hugging Face token in TOML format:
                   ```
                   HF_TOKEN = "your_token_here"
                   ```
                3. Make sure the token has quotes around it
                4. Click Save and wait for redeploy
                5. Check the Debug Info section in the sidebar for token status
                """)
            elif "quantization" in error_msg.lower() or "cpu" in error_msg.lower():
                st.warning("**Device/Quantization Error**")
                st.info("""
                This might be a CPU/GPU compatibility issue. The model will try to load without quantization.
                """)
            
            # Show full error in expander
            with st.expander("🔍 Full Error Details"):
                st.exception(e)
            
            raise
    
    return st.session_state.email_model


def main():
    # Initialize session state (must be at start of main function)
    if 'email_model' not in st.session_state:
        st.session_state.email_model = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Debug: Check if HF_TOKEN is available (only show in sidebar for debugging)
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    
    st.title("✉️ LoRA-Mail Assistant")
    st.markdown("Generate emails in your personal style using AI fine-tuned with LoRA")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        tone = st.selectbox(
            "Email Tone",
            ["professional", "casual", "friendly", "formal", "warm"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.text(f"Base Model: {config['model']['base_model']}")
        st.text(f"Device: {config['inference']['device']}")
        
        # Show model status
        if st.session_state.email_model is not None:
            st.success("✅ Model loaded and cached")
        else:
            st.info("ℹ️ Model will load on first generation")
        
        # Clear cache button
        if st.button("🔄 Reload Model", help="Clear cached model and reload (use if model seems stuck)"):
            st.session_state.email_model = None
            st.success("Model cache cleared. Will reload on next generation.")
            st.rerun()
        
        # Debug section (expandable)
        with st.expander("🔍 Debug Info (for troubleshooting)"):
            st.session_state.show_debug = True
            
            st.subheader("Token Status")
            # Check Streamlit secrets
            try:
                if hasattr(st, 'secrets'):
                    hf_token_secret = None
                    access_method = None
                    try:
                        hf_token_secret = st.secrets.get("HF_TOKEN")
                        access_method = "st.secrets.get()"
                    except Exception as e1:
                        try:
                            hf_token_secret = st.secrets["HF_TOKEN"]
                            access_method = "st.secrets[]"
                        except Exception as e2:
                            try:
                                hf_token_secret = getattr(st.secrets, "HF_TOKEN", None)
                                access_method = "getattr()"
                            except Exception as e3:
                                st.error(f"All access methods failed: {e1}, {e2}, {e3}")
                    
                    if hf_token_secret:
                        st.success(f"✅ HF_TOKEN found in secrets (via {access_method})")
                        st.code(f"Token length: {len(hf_token_secret)} characters")
                        st.code(f"Token starts with: {hf_token_secret[:10]}...")
                        st.code(f"Token ends with: ...{hf_token_secret[-5:]}")
                        
                        # Check if token looks valid
                        if hf_token_secret.startswith("hf_"):
                            st.success("✅ Token format looks correct (starts with 'hf_')")
                        else:
                            st.warning("⚠️ Token doesn't start with 'hf_' - might be invalid")
                    else:
                        st.error("❌ HF_TOKEN not found in Streamlit secrets")
                        st.info("**How to fix:**")
                        st.code('''# In Streamlit Cloud → Settings → Secrets, add:
HF_TOKEN = "your_token_here"''', language="toml")
                else:
                    st.warning("⚠️ st.secrets not available")
            except Exception as e:
                st.error(f"Error checking secrets: {str(e)}")
                st.exception(e)
            
            st.markdown("---")
            st.subheader("Environment Variables")
            # Check environment variables
            import os
            env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_API_TOKEN")
            if env_token:
                st.success(f"✅ HF_TOKEN found in environment (length: {len(env_token)})")
            else:
                st.info("ℹ️ HF_TOKEN not in environment variables (this is OK if using Streamlit secrets)")
            
            st.markdown("---")
            st.subheader("Model Configuration")
            st.code(f"Base Model: {config['model']['base_model']}")
            st.code(f"Device: {config['inference']['device']}")
            st.code(f"Use Quantization: {config['inference']['use_quantization']}")
            
            st.markdown("---")
            st.subheader("Model Status")
            if st.session_state.model_loaded:
                st.success("✅ Model is loaded and cached")
            else:
                st.info("ℹ️ Model will load on first generation")
        
        # Comparison option
        show_comparison = st.checkbox(
            "Show Base Model Comparison", 
            value=False, 
            help="Compare fine-tuned vs base model (slower, but shows the difference)"
        )
        
        # Mode selection
        use_api = st.checkbox("Use API Server (optional)", value=False, help="If checked, uses FastAPI server. Otherwise, uses model directly.")
        
        if use_api:
            if st.button("Check API Status"):
                try:
                    response = requests.get(f"{api_url}/health", timeout=2)
                    if response.status_code == 200:
                        st.success("✅ API is running")
                    else:
                        st.error("❌ API returned an error")
                except requests.exceptions.RequestException:
                    st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")
    
    # Main form
    with st.form("email_form"):
        st.subheader("Email Context")
        
        message_description = st.text_area(
            "Message Description",
            placeholder="Describe what you want to communicate in this email...",
            help="Describe the message you want to send. Include key points, purpose, and any important details.",
            height=150
        )
        
        submitted = st.form_submit_button("Generate Email", use_container_width=True)
    
    # Generate email
    if submitted:
        if not message_description:
            st.error("Please fill in the Message Description field.")
        else:
            # Use message description as purpose, empty key_points for simplified prompt
            purpose = message_description
            key_points = ""  # Empty to trigger simplified prompt format
            
            with st.spinner("Generating emails (this may take a moment)..."):
                try:
                    if use_api:
                        # Use API server
                        response = requests.post(
                            f"{api_url}/generate",
                            json={
                                "purpose": purpose,
                                "key_points": key_points,
                                "tone": tone
                            },
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            fine_tuned_email = result['email']
                            # API doesn't support base model comparison, so skip it
                            base_email = None
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                            st.stop()
                    else:
                        # Get model (will load only first time, then use cached)
                        import time
                        
                        # Check if model needs loading
                        load_start = time.time()
                        model = get_model()
                        load_time = time.time() - load_start
                        
                        # Only show if it actually loaded (took > 1 second)
                        if load_time > 1:
                            st.info(f"Model loaded in {load_time:.1f}s")
                        
                        # Generate from fine-tuned model
                        status_text = st.empty()
                        progress_bar = st.progress(0)
                        status_text.info("🔄 Generating email...")
                        progress_bar.progress(10)
                        
                        gen_start = time.time()
                        try:
                            # Generate with progress updates
                            progress_bar.progress(30)
                            fine_tuned_email = model.generate_email(
                                recipient="",  # Not needed - user only wants content
                                purpose=purpose,
                                key_points=key_points,
                                tone=tone
                            )
                            progress_bar.progress(90)
                            gen_time = time.time() - gen_start
                            
                            if gen_time > 60:
                                status_text.error(f"❌ Generation took {gen_time:.1f} seconds - this is too slow!")
                                st.error("Generation is taking too long. Try clicking 'Reload Model' button.")
                            elif gen_time > 30:
                                status_text.warning(f"⚠️ Generated in {gen_time:.1f} seconds (slower than expected)")
                            else:
                                status_text.success(f"✅ Generated in {gen_time:.1f} seconds")
                            
                            progress_bar.progress(100)
                            
                        except Exception as gen_error:
                            gen_time = time.time() - gen_start
                            status_text.error(f"❌ Generation failed after {gen_time:.1f} seconds: {str(gen_error)}")
                            progress_bar.empty()
                            raise gen_error
                        
                        base_email = None
                        
                        # Generate from base model only if comparison is enabled
                        if show_comparison:
                            status_text.info("🔄 Generating base model comparison...")
                            base_email = model.generate_email_base(
                                recipient="",  # Not needed - user only wants content
                                purpose=purpose,
                                key_points=key_points,
                                tone=tone
                            )
                        
                        status_text.empty()
                        progress_bar.empty()
                    
                    st.success("✅ Email generated successfully!")
                    
                    st.markdown("---")
                    
                    # Display based on whether comparison is enabled
                    if show_comparison and base_email:
                        st.subheader("📊 Model Comparison")
                        
                        # Display side-by-side comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### ✨ Fine-Tuned Model (Your Style)")
                            st.markdown("*Trained on your email writing style*")
                            st.text_area(
                                "Fine-Tuned Email",
                                value=fine_tuned_email,
                                height=400,
                                label_visibility="collapsed",
                                key="fine_tuned"
                            )
                            st.download_button(
                                label="📥 Download Fine-Tuned",
                                data=fine_tuned_email,
                                file_name="generated_email_finetuned.txt",
                                mime="text/plain",
                                key="download_finetuned"
                            )
                        
                        with col2:
                            st.markdown("### 🔵 Base Model (Generic)")
                            st.markdown("*No fine-tuning - generic output*")
                            st.text_area(
                                "Base Model Email",
                                value=base_email,
                                height=400,
                                label_visibility="collapsed",
                                key="base"
                            )
                            st.download_button(
                                label="📥 Download Base",
                                data=base_email,
                                file_name="generated_email_base.txt",
                                mime="text/plain",
                                key="download_base"
                            )
                    else:
                        # Single email display (faster)
                        st.subheader("✨ Generated Email (Fine-Tuned)")
                        st.markdown("*Trained on your email writing style*")
                        st.text_area(
                            "Email Content",
                            value=fine_tuned_email,
                            height=400,
                            label_visibility="collapsed",
                            key="email"
                        )
                        
                        st.download_button(
                            label="📥 Download Email",
                            data=fine_tuned_email,
                            file_name="generated_email.txt",
                            mime="text/plain",
                            key="download"
                        )
                        
                        if not show_comparison:
                            st.info("💡 Tip: Enable 'Show Base Model Comparison' in the sidebar to see the difference!")
                
                except requests.exceptions.RequestException as e:
                    st.error(f"Cannot connect to API. Make sure the FastAPI server is running at {api_url}")
                    st.code(f"Error: {str(e)}")
                except Exception as e:
                    st.error(f"Error generating email: {str(e)}")
                    st.code(f"Error details: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "LoRA-Mail Assistant - Fine-tuned for your writing style"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

