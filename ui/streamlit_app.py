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

# Initialize session state
if 'email_model' not in st.session_state:
    st.session_state.email_model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

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
            status.error(f"❌ Failed to load model: {str(e)}")
            raise
    
    return st.session_state.email_model


def main():
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
        
        recipient = st.text_input(
            "Recipient",
            placeholder="e.g., john.doe@company.com",
            help="Who is this email for?"
        )
        
        message_description = st.text_area(
            "Message Description",
            placeholder="Describe what you want to communicate in this email...",
            help="Describe the message you want to send. Include key points, purpose, and any important details.",
            height=150
        )
        
        submitted = st.form_submit_button("Generate Email", use_container_width=True)
    
    # Generate email
    if submitted:
        if not recipient or not message_description:
            st.error("Please fill in both Recipient and Message Description fields.")
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
                                "recipient": recipient,
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
                                recipient=recipient,
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
                                recipient=recipient,
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
                                file_name=f"email_finetuned_{recipient.replace('@', '_at_')}.txt",
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
                                file_name=f"email_base_{recipient.replace('@', '_at_')}.txt",
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
                            file_name=f"email_{recipient.replace('@', '_at_')}.txt",
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

