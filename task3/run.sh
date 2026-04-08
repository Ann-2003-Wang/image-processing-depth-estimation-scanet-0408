python /225040103/task3/train.py \
  --scannet_root /225040103/data/scannet \
  --output_dir /225040103/task3/baseline \
  --train_split_file /225040103/data/scannet/scannetv2_train.txt \
  --max_train_samples 10000 \
  --epochs 32 \
  --batch_size 128 \
  --num_workers 4

./test_baseline.sh


# 实验1：16轮，20000样本，输出至 20000_16_opt
python /225040103/task3/train.py \
  --scannet_root /225040103/data/scannet \
  --output_dir /225040103/t3.3/20000_16_opt \
  --train_split_file /225040103/data/scannet/scannetv2_train.txt \
  --max_train_samples 20000 \
  --epochs 16 \
  --batch_size 128 \
  --num_workers 4

/225040103/t3.3/test_2000016.sh


# 实验2：8轮，40000样本，输出至 40000_8_opt
python /225040103/task3/train.py \
  --scannet_root /225040103/data/scannet \
  --output_dir /225040103/t3.3/40000_8_opt \
  --train_split_file /225040103/data/scannet/scannetv2_train.txt \
  --max_train_samples 40000 \
  --epochs 8 \
  --batch_size 128 \
  --num_workers 4\
  --resume /225040103/t3.3/40000_8_opt/epoch_001.pth
/225040103/t3.3/test_400008.sh



# 实验3：4轮，80000样本，输出至 80000_4_opt
python /225040103/task3/train.py \
  --scannet_root /225040103/data/scannet \
  --output_dir /225040103/t3.3/80000_4_opt \
  --train_split_file /225040103/data/scannet/scannetv2_train.txt \
  --max_train_samples 80000 \
  --epochs 4 \
  --batch_size 128 \
  --num_workers 4
/225040103/t3.3/test_800004.sh


# 实验4：2轮，160000样本，输出至 160000_2_opt
python /225040103/task3/train.py \
  --scannet_root /225040103/data/scannet \
  --output_dir /225040103/t3.3/160000_2_opt \
  --train_split_file /225040103/data/scannet/scannetv2_train.txt \
  --max_train_samples 160000 \
  --epochs 2 \
  --batch_size 128 \
  --num_workers 4
  
/225040103/t3.3/test_1600002.sh
