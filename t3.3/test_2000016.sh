#!/bin/bash


#赋予脚本执行权限：chmod +x test_2000016.sh
#运行：./test_2000016.sh


# 请根据实际实验修改以下路径和范围
OUTPUT_DIR="/225040103/t3.3/20000_16_opt"   # 模型与结果存放目录
EPOCH_START=001
EPOCH_END=016

mkdir -p ${OUTPUT_DIR}/val_results
cd /225040103/task3   # 确保 test.py 所在目录正确

# 1. 测试所有 epoch
for epoch in $(seq -f "%03g" ${EPOCH_START} ${EPOCH_END}); do
    echo "Testing epoch_${epoch}.pth ..."
    python test.py \
        --scannet_root /225040103/data/scannet \
        --split_file /225040103/data/scannet/scannetv2_val.txt \
        --checkpoint ${OUTPUT_DIR}/epoch_${epoch}.pth \
        --save_json ${OUTPUT_DIR}/val_results/epoch_${epoch}.json \
        --batch_size 128 \
        --num_workers 4
done

# 2. 找出最小 AbsRel 对应的 epoch（只输出数字）
best_epoch=$(python -c "
import json, glob, os
best = float('inf')
best_epoch = None
for f in glob.glob('${OUTPUT_DIR}/val_results/epoch_*.json'):
    with open(f) as fp:
        data = json.load(fp)
        abs_rel = data['abs_rel']
        epoch_num = int(os.path.basename(f).split('_')[1].split('.')[0])
        if abs_rel < best:
            best = abs_rel
            best_epoch = epoch_num
print(best_epoch)
")

# 格式化为三位数，用于文件名
best_epoch_padded=$(printf "%03d" ${best_epoch})
echo "最佳验证 AbsRel 对应的 epoch 为: ${best_epoch_padded}"

# 3. 对最佳 epoch 再次运行测试（生成可视化图片）
echo "Generating visualizations for best epoch ${best_epoch_padded} ..."
python test.py \
    --scannet_root /225040103/data/scannet \
    --split_file /225040103/data/scannet/scannetv2_val.txt \
    --checkpoint ${OUTPUT_DIR}/epoch_${best_epoch_padded}.pth \
    --save_json ${OUTPUT_DIR}/val_results/epoch_${best_epoch}_best.json \
    --batch_size 128 \
    --num_workers 4

echo "可视化图片已保存至: ${OUTPUT_DIR}/visualizations/"

# 4. 生成表格和曲线（过滤 _best 文件）
python -c "
import json, glob, os
import pandas as pd
import matplotlib.pyplot as plt

data = []
for f in glob.glob('${OUTPUT_DIR}/val_results/epoch_*.json'):
    if '_best' in os.path.basename(f):
        continue
    epoch = int(os.path.basename(f).split('_')[1].split('.')[0])
    with open(f) as fp:
        abs_rel = json.load(fp)['abs_rel']
    data.append((epoch, abs_rel))

data.sort(key=lambda x: x[0])
df = pd.DataFrame(data, columns=['epoch', 'abs_rel'])
df.to_csv('${OUTPUT_DIR}/val_results/performance.csv', index=False)

plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['abs_rel'], marker='o')
plt.xlabel('Epoch')
plt.ylabel('AbsRel')
plt.title('Validation AbsRel vs. Epoch')
plt.grid(True)
plt.savefig('${OUTPUT_DIR}/val_results/performance_curve.png')
print('表格和曲线已保存至:', '${OUTPUT_DIR}/val_results/')
"
