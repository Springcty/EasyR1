import modal
import os

# 1. Define Image (Same as before)
image = (
    modal.Image.from_registry("tianyuc/easyr1-seedbench:v1")
    .pip_install("wandb")
    .add_local_dir(
        "/home/tianyuca/EasyR1", 
        remote_path="/root/EasyR1",
        ignore=["logs", "wandb", ".git", ".gitignore"]
    )
)

app = modal.App("easyr1-qwen3-vl-2b-seedr1-grpo")

# 2. Define Volumes
# Connect to your EXISTING data volume
data_vol = modal.Volume.from_name("easyr1-seed-bench-r1-data", create_if_missing=False)

# Create NEW volumes for persistence
ckpt_vol = modal.Volume.from_name("easyr1-checkpoints", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("easyr1-hf-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100:8",
    timeout=86400,
    # 3. Mount Volumes
    volumes={
        "/root/EasyR1/data": data_vol,           # Mounts your 'RL', 'images', 'videos' to data/
        "/root/EasyR1/outputs": ckpt_vol,        # Saves training outputs/checkpoints here
        "/root/.cache/huggingface": hf_cache_vol # Caches downloaded HF models here
    },
    secrets=[
        modal.Secret.from_name("wandb-secret-tianyu"),
        modal.Secret.from_name("hf-secret-tianyu"),
    ]
)
def train():
    import subprocess
    os.chdir("/root/EasyR1")
    
    # 4. Debug: Verify data is visible
    print("Checking data volume...")
    subprocess.run("ls -F data/", shell=True) 
    
    script_path = "examples/modal_scripts/qwen3_vl_2b_seedr1_grpo_default.sh"
    subprocess.run(["chmod", "+x", script_path], check=True)
    
    print(f"Starting training: {script_path}")
    subprocess.run(f"bash {script_path}", shell=True, check=True)

if __name__ == "__main__":
    with app.run():
        train.remote()