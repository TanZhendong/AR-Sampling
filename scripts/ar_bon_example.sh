export CUDA_VISIBLE_DEVICES="0"
export HF_HOME="XXX"
export HF_ENDPOINT=https://hf-mirror.com

ar_bon() {
    python test_time_compute.py configs/ar_bon.yaml \
    --output_dir=$1
}

# for example
ar_bon "outputs/math500/ar_bon"