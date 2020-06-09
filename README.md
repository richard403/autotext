# autotext

### 终端添加python环境
```angular2
export PYTHONPATH="/home/whx/workspace/work/python_code/autotext:{$PYTHONPATH}"
```

### 开启可视化
```angular2
tensorboard --logdir=./data/tensorboard/rnn/20200609_11 --bind_all
```

### 训练文件转成TFRecord
```angular2
python ./datamanager/dataload.py
```

### 训练
```angular2
python ./rnn/train.py --num_epochs=600 --restore --batch_size=50  --mode=train --tensorboard
```

### 测试
```angular2
python ./rnn/train.py --test_num 200 --load_flag 20200608_17   --mode=test
```