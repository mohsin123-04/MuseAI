"""
MuseAI Terminal-Based Command Line Interface
============================================

Simple terminal-based interface for neural style transfer.

Usage:
    python cli.py                    # Start interactive menu
    python cli.py --help             # Show help
    python cli.py --upload-dir DIR   # Set upload directory

Directory Structure:
    uploads/
    ├── user_1/
    │   ├── input/
    │   │   └── selfie.jpg
    │   └── output/
    │       ├── picasso_output.png
    │       └── rembrandt_output.png
    ├── user_2/
    └── ...
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import time

# Import MuseAI modules
try:
    from src.inference import StyleTransferInference
    from src.config import get_device, IMAGE_SIZE
    import torch
except ImportError as e:
    print(f"❌ Error: Missing required modules. Run: pip install -r requirements.txt")
    print(f"   Details: {e}")
    sys.exit(1)


class MuseAICLI:
    """Terminal-based interface for MuseAI style transfer"""
    
    def __init__(self, upload_base_dir="uploads"):
        """
        Initialize CLI
        
        Args:
            upload_base_dir: Base directory for user uploads
        """
        self.upload_base_dir = Path(upload_base_dir)
        self.upload_base_dir.mkdir(exist_ok=True)
        
        # Create uploads folder structure
        self._setup_uploads_folder()
        
        # Load inference model
        print("🔄 Loading MuseAI model...")
        try:
            self.model = StyleTransferInference.load_best_checkpoint()
            print("✅ Model loaded successfully!")
            device = get_device()
            print(f"✅ Using device: {device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"   Make sure checkpoint files exist in checkpoints/")
            sys.exit(1)
        
        self.artists = ["picasso", "rembrandt"]
        self.current_user = None
        self.current_image = None
        
    def _setup_uploads_folder(self):
        """Create base uploads folder structure"""
        self.upload_base_dir.mkdir(exist_ok=True, parents=True)
        print(f"📁 Uploads folder: {self.upload_base_dir.absolute()}")
    
    def _get_user_folders(self, user_id):
        """Get or create user-specific folders"""
        user_dir = self.upload_base_dir / user_id
        user_dir.mkdir(exist_ok=True, parents=True)
        
        input_dir = user_dir / "input"
        output_dir = user_dir / "output"
        
        input_dir.mkdir(exist_ok=True, parents=True)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        return {
            "user": user_dir,
            "input": input_dir,
            "output": output_dir
        }
    
    def print_header(self):
        """Print application header"""
        print("\n" + "="*60)
        print("🎨 MuseAI - Neural Style Transfer (Terminal Version)")
        print("="*60)
        print("Transform your portrait into Picasso or Rembrandt style!")
        print()
    
    def print_menu(self):
        """Print main menu"""
        print("\n" + "-"*60)
        print("📋 MAIN MENU")
        print("-"*60)
        print("1. 🆕 Create New User / Select User")
        print("2. 📤 Upload Selfie")
        print("3. 🎨 Apply Style Transfer")
        print("4. 📂 List Users & Images")
        print("5. 💾 View Output Images")
        print("6. 🗑️  Delete User Data")
        print("7. ❓ Help & Commands")
        print("0. ❌ Exit")
        print("-"*60)
    
    def create_user(self):
        """Create new user or select existing user"""
        print("\n📝 USER SELECTION")
        print("-"*60)
        
        # List existing users
        existing_users = self._list_existing_users()
        
        if existing_users:
            print("\n📋 Existing Users:")
            for i, user in enumerate(existing_users, 1):
                print(f"   {i}. {user}")
            print(f"   {len(existing_users) + 1}. Create New User")
            
            try:
                choice = input("\nSelect user number (or 'n' for new): ").strip()
                
                if choice.lower() == 'n' or choice == str(len(existing_users) + 1):
                    user_id = self._prompt_new_user_id()
                else:
                    idx = int(choice) - 1
                    if 0 <= idx < len(existing_users):
                        user_id = existing_users[idx]
                    else:
                        print("❌ Invalid selection")
                        return
            except ValueError:
                print("❌ Invalid input")
                return
        else:
            print("No existing users found. Create a new user.")
            user_id = self._prompt_new_user_id()
        
        # Set current user
        self.current_user = user_id
        folders = self._get_user_folders(user_id)
        
        print(f"\n✅ User set to: {user_id}")
        print(f"   Input folder:  {folders['input']}")
        print(f"   Output folder: {folders['output']}")
    
    def _prompt_new_user_id(self):
        """Prompt for new user ID"""
        while True:
            user_id = input("\nEnter user ID (alphanumeric, no spaces): ").strip()
            
            if not user_id:
                print("❌ User ID cannot be empty")
                continue
            
            if not user_id.replace('_', '').replace('-', '').isalnum():
                print("❌ Use only letters, numbers, underscores, or hyphens")
                continue
            
            return user_id
    
    def _list_existing_users(self):
        """List all existing users"""
        if not self.upload_base_dir.exists():
            return []
        
        users = [d.name for d in self.upload_base_dir.iterdir() 
                if d.is_dir()]
        return sorted(users)
    
    def upload_selfie(self):
        """Upload selfie to current user's input folder"""
        if not self.current_user:
            print("❌ Please select a user first (option 1)")
            return
        
        print("\n📤 UPLOAD SELFIE")
        print("-"*60)
        
        folders = self._get_user_folders(self.current_user)
        input_folder = folders['input']
        
        print(f"Input folder: {input_folder}")
        print("\nOptions:")
        print("1. Copy image file")
        print("2. Paste from clipboard path")
        print("3. Cancel")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            self._copy_image_file(input_folder)
        elif choice == "2":
            self._paste_image_path(input_folder)
        elif choice == "3":
            print("Cancelled")
        else:
            print("❌ Invalid choice")
    
    def _copy_image_file(self, input_folder):
        """Copy image file to input folder"""
        image_path = input("\nEnter full path to image (e.g., C:\\Users\\Photo\\selfie.jpg): ").strip()
        
        image_path = image_path.strip('"')  # Remove quotes if present
        
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return
        
        # Validate image extension
        valid_ext = ['.jpg', '.jpeg', '.png', '.bmp']
        if not any(image_path.lower().endswith(ext) for ext in valid_ext):
            print("❌ Unsupported image format. Use JPG, PNG, or BMP")
            return
        
        try:
            filename = Path(image_path).name
            dest_path = input_folder / filename
            
            print(f"📋 Copying: {filename}")
            shutil.copy2(image_path, dest_path)
            self.current_image = filename
            print(f"✅ Image uploaded: {dest_path}")
            print(f"   File size: {os.path.getsize(dest_path) / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"❌ Error copying file: {e}")
    
    def _paste_image_path(self, input_folder):
        """User provides image path directly"""
        image_path = input("\nPaste image path here: ").strip().strip('"')
        self._copy_image_file(input_folder)
    
    def list_users_and_images(self):
        """List all users and their uploaded images"""
        print("\n📂 USERS & IMAGES")
        print("-"*60)
        
        users = self._list_existing_users()
        
        if not users:
            print("No users found")
            return
        
        for user_id in users:
            folders = self._get_user_folders(user_id)
            input_folder = folders['input']
            output_folder = folders['output']
            
            print(f"\n👤 User: {user_id}")
            
            # List input images
            input_files = list(input_folder.glob("*"))
            if input_files:
                print(f"   📥 Input ({len(input_files)}):")
                for f in input_files:
                    size = os.path.getsize(f) / 1024
                    print(f"      • {f.name} ({size:.1f} KB)")
            else:
                print(f"   📥 Input: (empty)")
            
            # List output images
            output_files = list(output_folder.glob("*.png"))
            if output_files:
                print(f"   📤 Output ({len(output_files)}):")
                for f in output_files:
                    size = os.path.getsize(f) / 1024 / 1024
                    print(f"      • {f.name} ({size:.2f} MB)")
            else:
                print(f"   📤 Output: (empty)")
    
    def apply_style_transfer(self):
        """Apply style transfer to uploaded image"""
        if not self.current_user:
            print("❌ Please select a user first (option 1)")
            return
        
        folders = self._get_user_folders(self.current_user)
        input_folder = folders['input']
        output_folder = folders['output']
        
        # Find input image
        input_files = list(input_folder.glob("*"))
        input_files = [f for f in input_files if f.suffix.lower() in 
                      ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not input_files:
            print("❌ No image found in input folder")
            print(f"   Please upload an image first (option 2)")
            return
        
        # Select image if multiple
        if len(input_files) > 1:
            print("\n📸 Multiple images found:")
            for i, f in enumerate(input_files, 1):
                print(f"   {i}. {f.name}")
            
            try:
                choice = int(input("Select image number: ")) - 1
                if 0 <= choice < len(input_files):
                    input_image = input_files[choice]
                else:
                    print("❌ Invalid selection")
                    return
            except ValueError:
                print("❌ Invalid input")
                return
        else:
            input_image = input_files[0]
        
        print("\n🎨 STYLE TRANSFER")
        print("-"*60)
        print(f"Input image: {input_image.name}")
        print("\nAvailable artists:")
        
        for i, artist in enumerate(self.artists, 1):
            print(f"  {i}. {artist.capitalize()}")
        
        # Select artist
        try:
            artist_choice = int(input("\nSelect artist (1 or 2): ")) - 1
            if 0 <= artist_choice < len(self.artists):
                artist = self.artists[artist_choice]
            else:
                print("❌ Invalid selection")
                return
        except ValueError:
            print("❌ Invalid input")
            return
        
        # Style strength
        try:
            style_str = input("Style strength (0-1.0, default 1.0): ").strip()
            style_strength = float(style_str) if style_str else 1.0
            
            if not 0 <= style_strength <= 1.0:
                print("❌ Style strength must be between 0 and 1.0")
                return
        except ValueError:
            print("❌ Invalid style strength")
            return
        
        # Process image
        print(f"\n⏳ Processing: {artist.capitalize()} style (strength: {style_strength})")
        print("   This may take 30-60 seconds...")
        
        try:
            start_time = time.time()
            
            # Run style transfer
            output_image = self.model.stylize_selfie(
                selfie_path=str(input_image),
                artist=artist,
                style_strength=style_strength
            )
            
            # Save output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"{artist}_{timestamp}.png"
            output_path = output_folder / output_name
            
            output_image.save(str(output_path))
            
            elapsed = time.time() - start_time
            file_size = os.path.getsize(output_path) / 1024 / 1024
            
            print(f"\n✅ Style transfer complete!")
            print(f"   Output: {output_path}")
            print(f"   Size: {file_size:.2f} MB")
            print(f"   Time: {elapsed:.1f} seconds")
            
        except Exception as e:
            print(f"❌ Error during style transfer: {e}")
            import traceback
            traceback.print_exc()
    
    def view_output_images(self):
        """Open output folder for viewing images"""
        if not self.current_user:
            print("❌ Please select a user first (option 1)")
            return
        
        folders = self._get_user_folders(self.current_user)
        output_folder = folders['output']
        
        output_files = list(output_folder.glob("*.png"))
        
        if not output_files:
            print("❌ No output images found")
            return
        
        print("\n📸 OUTPUT IMAGES")
        print("-"*60)
        
        for i, f in enumerate(output_files, 1):
            size = os.path.getsize(f) / 1024 / 1024
            created = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"{i}. {f.name}")
            print(f"   Size: {size:.2f} MB | Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📂 Output folder: {output_folder}")
        print("Use File Explorer or terminal to view/copy images")
    
    def delete_user_data(self):
        """Delete user and all associated data"""
        users = self._list_existing_users()
        
        if not users:
            print("❌ No users found")
            return
        
        print("\n🗑️  DELETE USER DATA")
        print("-"*60)
        print("Existing users:")
        
        for i, user in enumerate(users, 1):
            print(f"   {i}. {user}")
        
        try:
            choice = int(input("\nSelect user to delete (0 to cancel): ")) - 1
            
            if choice == -1:
                return
            
            if 0 <= choice < len(users):
                user_id = users[choice]
                confirm = input(f"Delete all data for '{user_id}'? (yes/no): ").strip().lower()
                
                if confirm == 'yes':
                    user_dir = self.upload_base_dir / user_id
                    shutil.rmtree(user_dir)
                    
                    if self.current_user == user_id:
                        self.current_user = None
                    
                    print(f"✅ User '{user_id}' deleted")
                else:
                    print("Cancelled")
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Invalid input")
    
    def show_help(self):
        """Show help and example commands"""
        print("\n❓ HELP & COMMANDS")
        print("="*60)
        print("""
📖 HOW TO USE MUSEAI CLI:

STEP 1: Create/Select User
   └─ Option 1: Create new user or select existing

STEP 2: Upload Image
   └─ Option 2: Upload selfie to user's input folder

STEP 3: Apply Style Transfer
   └─ Option 3: Transform image to Picasso or Rembrandt style

STEP 4: View Results
   └─ Option 5: View output images in user's folder

═══════════════════════════════════════════════════════════

🎨 EXAMPLE WORKFLOW:

User: john_doe
├─ Input: john_doe/input/selfie.jpg
└─ Output:
   ├─ picasso_20250605_143022.png (Picasso style)
   └─ rembrandt_20250605_143547.png (Rembrandt style)

═══════════════════════════════════════════════════════════

⚡ QUICK COMMANDS:

1️⃣  First time:
    • Select user (new)
    • Upload selfie
    • Apply Picasso style
    • Check output folder

2️⃣  Try another artist:
    • Apply Rembrandt style to same image

3️⃣  Try different style strength:
    • 0.3 = Subtle artistic touches
    • 0.6 = Balanced content & style
    • 1.0 = Maximum artistic style

═══════════════════════════════════════════════════════════

📂 FOLDER STRUCTURE:

uploads/
├── user_1/
│   ├── input/         (Upload selfies here)
│   └── output/        (Results saved here)
├── user_2/
└── ...

═══════════════════════════════════════════════════════════

🖥️  SYSTEM INFO:

• GPU Support: Yes (CUDA 12.5+)
• Input formats: JPG, PNG, BMP
• Output format: PNG (512×512)
• Processing time: ~30-60 seconds per image
• Identity preservation: Enabled (preserves facial features)

═══════════════════════════════════════════════════════════

❓ TROUBLESHOOTING:

Q: "Model loading failed"
A: Check that checkpoints/best_model.pth exists

Q: "No faces detected"
A: Use clear portrait photo with visible face

Q: "Slow processing"
A: Normal - GPU is computing style transfer

Q: "Where are my images?"
A: Check uploads/<your_user>/output/

═══════════════════════════════════════════════════════════
""")
    
    def run(self):
        """Main CLI loop"""
        self.print_header()
        
        try:
            while True:
                self.print_menu()
                
                choice = input("\nEnter choice (0-7): ").strip()
                
                if choice == "1":
                    self.create_user()
                elif choice == "2":
                    self.upload_selfie()
                elif choice == "3":
                    self.apply_style_transfer()
                elif choice == "4":
                    self.list_users_and_images()
                elif choice == "5":
                    self.view_output_images()
                elif choice == "6":
                    self.delete_user_data()
                elif choice == "7":
                    self.show_help()
                elif choice == "0":
                    print("\n👋 Thank you for using MuseAI!")
                    print("   Your images are saved in: uploads/")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Exiting MuseAI...")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MuseAI Terminal-Based CLI for Style Transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py                      # Start interactive menu
  python cli.py --help               # Show this help
  python cli.py --upload-dir custom  # Use custom uploads folder
        """
    )
    
    parser.add_argument(
        '--upload-dir',
        default='uploads',
        help='Base directory for user uploads (default: uploads/)'
    )
    
    args = parser.parse_args()
    
    # Initialize and run CLI
    cli = MuseAICLI(upload_base_dir=args.upload_dir)
    cli.run()


if __name__ == "__main__":
    main()
