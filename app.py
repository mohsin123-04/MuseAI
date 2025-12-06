"""
Gradio Web Interface for MuseAI
User-friendly web UI for uploading portraits and selecting artists.
"""
import gradio as gr
import torch
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import time

import sys
sys.path.append(str(Path(__file__).parent))

from src.inference import StyleTransferInference
from src.config import CHECKPOINTS_DIR, ARTISTS, IMAGE_SIZE


class MuseAIInterface:
    """Gradio interface wrapper for MuseAI."""
    
    def __init__(self):
        """Initialize the interface and load model."""
        print("Initializing MuseAI Interface...")
        
        # Load best checkpoint or untrained model
        self.model = StyleTransferInference.load_best_checkpoint()
        print("Model loaded successfully")
        
        # Cache for generated images
        self.last_stylized = None
        self.last_artist = None
    
    def stylize(
        self,
        portrait_image: np.ndarray,
        artist: str,
        style_strength: float = 1.0
    ) -> Tuple[Image.Image, str]:
        """
        Apply style transfer to a portrait.
        
        Args:
            portrait_image: Input image as numpy array (from Gradio)
            artist: Selected artist ('picasso' or 'rembrandt')
            style_strength: Style strength (0.0-1.0)
        
        Returns:
            Tuple of (stylized image, status message)
        """
        try:
            # Validate artist
            if artist not in ARTISTS:
                return None, f"❌ Invalid artist: {artist}. Choose from: {', '.join(ARTISTS)}"
            
            # Validate style strength
            if not (0.0 <= style_strength <= 1.0):
                return None, f"❌ Style strength must be between 0.0 and 1.0"
            
            # Convert numpy array to PIL Image and save temporarily
            portrait_pil = Image.fromarray(portrait_image.astype('uint8'))
            temp_path = Path("/tmp/muse_input.jpg")
            portrait_pil.save(temp_path)
            
            # Process
            print(f"\nProcessing portrait in {artist.upper()} style...")
            start_time = time.time()
            
            stylized = self.model.stylize_selfie(
                selfie_path=temp_path,
                artist=artist,
                style_strength=style_strength,
                return_tensor=False
            )
            
            elapsed = time.time() - start_time
            
            # Cache result
            self.last_stylized = stylized
            self.last_artist = artist
            
            # Status message
            face_status = "✅ Face detected and processed"
            message = f"{face_status}\n⏱️ Processing time: {elapsed:.2f}s\n🎨 Artist: {artist.upper()}\n💪 Style strength: {style_strength:.1%}"
            
            return stylized, message
            
        except Exception as e:
            print(f"Error during stylization: {e}")
            import traceback
            traceback.print_exc()
            return None, f"❌ Error: {str(e)}"
    
    def download_image(self) -> Tuple[str, str]:
        """
        Get the last stylized image for download.
        
        Returns:
            Path and filename for download
        """
        if self.last_stylized is None:
            return None, "❌ No image generated yet. Please stylize a portrait first."
        
        # Save image
        output_path = Path(f"/tmp/muse_output_{self.last_artist}.png")
        self.last_stylized.save(output_path)
        
        return str(output_path), f"✅ Image ready: {output_path.name}"
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        
        with gr.Blocks(
            title="MuseAI - Neural Style Transfer",
            theme=gr.themes.Soft()
        ) as demo:
            
            gr.Markdown("# 🎨 MuseAI - Portrait Style Transfer")
            gr.Markdown(
                """
                Transform your portrait into artistic styles inspired by famous artists.
                
                **How to use:**
                1. 📸 Upload or take a portrait photo
                2. 🎭 Select an artist (Picasso or Rembrandt)
                3. 🎚️ Adjust style strength (0% = original, 100% = full style)
                4. ✨ Click "Generate" to transform your portrait
                5. 📥 Download your stylized portrait
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Portrait")
                    portrait_input = gr.Image(
                        label="Upload Portrait",
                        type="numpy",
                        sources=["upload", "webcam"],
                        scale=1
                    )
                    
                    gr.Markdown("### Settings")
                    artist_choice = gr.Radio(
                        choices=ARTISTS,
                        value=ARTISTS[0],
                        label="🎭 Select Artist",
                        info="Choose which artist style to apply"
                    )
                    
                    style_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.1,
                        label="💪 Style Strength",
                        info="0% = Original Portrait, 100% = Full Style"
                    )
                    
                    generate_btn = gr.Button(
                        "✨ Generate Styled Portrait",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Result")
                    portrait_output = gr.Image(
                        label="Stylized Portrait",
                        type="pil"
                    )
                    
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=4
                    )
                    
                    download_btn = gr.Button(
                        "📥 Download Image",
                        variant="secondary"
                    )
                    
                    download_status = gr.Textbox(
                        label="Download Status",
                        interactive=False
                    )
            
            # Event handlers
            generate_btn.click(
                fn=self.stylize,
                inputs=[portrait_input, artist_choice, style_strength],
                outputs=[portrait_output, status_output]
            )
            
            download_btn.click(
                fn=self.download_image,
                outputs=[gr.File(), download_status]
            )
            
            # Examples
            gr.Markdown("### 💡 Tips")
            gr.Markdown(
                """
                - **Best Results:** Use well-lit portrait photos with clear facial features
                - **Picasso Style:** Cubist, geometric, bold colors
                - **Rembrandt Style:** Classical oil painting, dramatic lighting
                - **Style Strength:** 
                  - 30-50%: Subtle artistic touches
                  - 50-80%: Balanced content & style
                  - 80-100%: Maximum artistic transformation
                """
            )
            
            # Footer
            gr.Markdown("---")
            gr.Markdown(
                """
                <div style="text-align: center">
                    <p><b>MuseAI MVP</b> - Identity-Preserving Neural Style Transfer</p>
                    <p>Powered by PyTorch, VGG19 Encoder, AdaIN, and FaceNet</p>
                </div>
                """
            )
        
        return demo


def launch_interface(
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860
):
    """
    Launch the Gradio interface.
    
    Args:
        share: Whether to generate public shareable link
        server_name: Server address
        server_port: Server port
    """
    interface = MuseAIInterface()
    demo = interface.create_interface()
    
    print("\n" + "=" * 60)
    print("🚀 MuseAI Web Interface")
    print("=" * 60)
    print(f"Opening interface at http://localhost:{server_port}")
    if share:
        print("Share link will be generated (public access)")
    print("=" * 60 + "\n")
    
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True
    )


if __name__ == "__main__":
    # Launch with public share link
    launch_interface(share=False)
