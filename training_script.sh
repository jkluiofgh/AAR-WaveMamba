#!/bin/bash

# 转换格式 (防止 Windows 换行符报错)
# dos2unix training_script.sh

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

# === 配置区域 ===
Net='MambaWithDWT11'
Baseline='onlymamba'  # Baseline 模型名称 (需确保 models/onlymamba.py 存在)
Train_Path=../
Data_Path=/home/wangdongfang/work/mamba/0000PatchTST-TFC/CMI-Net/data/数据集/Goats/mydata/25hz*2s_3轴 
Save_Path=/home/wangdongfang/work/mamba/0000PatchTST-TFC/CMI-Net/checkpoint/MambaWithDWT11

# === 自动生成实验序号 ===
# 查找 Save_Path 下最大的数字文件夹，加 1 作为本次实验序号
last_num=$(ls -d "$Save_Path"/*/ 2>/dev/null | xargs -n 1 basename | sed 's/^\([0-9]\+\).*/\1/' | sort -n | tail -1)

if [ -z "$last_num" ]; then
  i=1
else
  i=$((last_num + 1))
fi

experiment_folder="$Save_Path/$i"
echo "当前实验序号是: $i"
echo "本次实验数据将保存在: $experiment_folder"

# 确保目标文件夹被创建
mkdir -p "$experiment_folder"

# === 定义训练函数 (三步走: Baseline -> DWT -> WPD/Ours) ===
# 参数1: 数据的ID (如 001)
# 参数2: 保存的子文件夹名 (如 1)
run_fold() {
    id=$1
    folder=$2
    
    data_file="$Data_Path/${id}goat.pt"
    
    # 路径定义
    main_save="$experiment_folder/$folder"          # Ours (WPD) 保存路径
    base_save="$main_save/baseline"                 # Baseline 保存路径
    dwt_save="$main_save/dwt_variant"               # DWT 对照组保存路径
    
    echo "========================================================"
    echo "Processing Fold: $id (Save to $folder)"
    
    # --- Step 1: 训练 Baseline (OnlyMamba) ---
    # 用于 t-SNE 展示 "无频域信息" 的状态
    echo ">> [Step 1/3] Training Baseline ($Baseline)..."
    uv run $Train_Path/train_wave2.py \
        --net "$Baseline" \
        --data_path "$data_file" \
        --save_path "$base_save" \
        --epoch 100 \
        --gpu 0  # 显式指定 GPU，根据需要修改
        
    base_ckpt="$base_save/${Baseline}-best.pth"
    
    # --- Step 2: 训练 DWT Variant (对照组) ---
    # 用于 t-SNE 展示 "仅低频分解" 的状态 (DWT vs WPD)
    echo ">> [Step 2/3] Training DWT Variant (Contrast Experiment)..."
    uv run $Train_Path/train_wave2.py \
        --net "$Net" \
        --decomp_type "dwt" \
        --data_path "$data_file" \
        --save_path "$dwt_save" \
        --epoch 100 \
        --gpu 0
        
    dwt_ckpt="$dwt_save/${Net}-best.pth"
    
    # --- Step 3: 训练 WPD Model (Ours) 并生成所有图表 ---
    # 传入 baseline_ckpt 和 dwt_ckpt，一次性画出所有对比图
    echo ">> [Step 3/3] Training Main WPD Model (Ours) & Generating Plots..."
    uv run $Train_Path/train_wave2.py \
        --net "$Net" \
        --decomp_type "wpd" \
        --data_path "$data_file" \
        --save_path "$main_save" \
        --baseline_ckpt "$base_ckpt" \
        --dwt_ckpt "$dwt_ckpt" \
        --epoch 200 \
        --gpu 0
}

# === 执行五折训练 ===
# 格式: run_fold "文件名编号" "保存文件夹名"

run_fold "001" "1"
run_fold "004" "4"
run_fold "002" "2"
run_fold "003" "3"
run_fold "005" "5"

# === 汇总结果 ===
echo "========================================================"
echo "All folds finished. Running Summary..."
# 注意：这里使用的是 五折汇总1.py (你提供的文件名)
uv run $Train_Path/五折汇总1.py --save_path "$experiment_folder"

echo "Done! Experiment saved to: $experiment_folder"