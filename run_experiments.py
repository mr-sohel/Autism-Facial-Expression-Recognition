import os
import subprocess
import time
import sys

# Force UTF-8 encoding for standard output to avoid UnicodeEncodeError with emojis on Windows
sys.stdout.reconfigure(encoding="utf-8")

def run_experiment(model_name, loss_type="ce_smooth", epochs=80, batch_size=16):
    print(f"\n{'='*50}")
    print(f"🚀 STARTING TRAINING: {model_name} (Loss: {loss_type})")
    print(f"{'='*50}\n")
    
    cmd = [
        "python", "src/train.py",
        "--model", model_name,
        "--loss", loss_type,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--mixup", 
        "--ema"
    ]
    
    start_time = time.time()
    # Using subprocess.run ensures we wait for it to finish before starting the next
    process = subprocess.run(cmd)
    
    elapsed = time.time() - start_time
    print(f"\n✅ FINISHED: {model_name} in {elapsed/60:.1f} minutes.\n")
    
    if process.returncode != 0:
        print(f"❌ ERROR: {model_name} failed with exit code {process.returncode}")
        return False
    return True

if __name__ == "__main__":
    print("Beginning Automated Training Pipeline...")
    
    # We use ce_smooth (Label Smoothing) for the baselines as standard.
    
    experiments = [
        # --- CNN Models ---
        {"model": "vgg16", "loss": "ce_smooth"},
        {"model": "vgg19", "loss": "ce_smooth"},
        {"model": "mobilenetv2_100", "loss": "ce_smooth"},
        {"model": "mobilenetv3_large_100", "loss": "ce_smooth"},
        {"model": "inception_v3", "loss": "ce_smooth"},
        {"model": "tf_efficientnetv2_s", "loss": "ce_smooth"},
        {"model": "tf_efficientnetv2_m", "loss": "ce_smooth"},
        {"model": "resnet50", "loss": "ce_smooth"},
        {"model": "densenet121", "loss": "ce_smooth"},
        {"model": "convnext_small", "loss": "ce_smooth"},
        
        # --- Transformer and Hybrid Models ---
        {"model": "vit_base_patch16_224", "loss": "ce_smooth"},
        {"model": "swin_base_patch4_window7_224", "loss": "ce_smooth"},
        {"model": "cvt_13", "loss": "ce_smooth"}, # Convolutional Vision Transformer (hybrid)
        {"model": "crossvit_9_240", "loss": "ce_smooth"}, # CrossViT (hybrid)
    ]
    
    for exp in experiments:
        success = run_experiment(exp["model"], loss_type=exp["loss"])
        if not success:
            print("Pipeline halted due to error.")
            break
            
    print("\n🎉 ALL EXPERIMENTS COMPLETED!")
    print("Check the results/logs/ and results/plots/ folders for your metrics and graphs!")
