export CUDA_VISIBLE_DEVICES="0"
export HF_HOME="XXX"
export HF_ENDPOINT=https://hf-mirror.com

best_of_n () {
    python test_time_compute.py configs/best_of_n.yaml \
    --output_dir=$1 \
    --dataset_name=$2 \
    --dataset_split=$3
}

# LLAMA3.2-1B
best_of_n "outputs/gsm8k/bon_16" "openai/gsm8k" "test"
