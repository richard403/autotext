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

### serving
```
docker run --rm -p 8501:8501 --mount type=bind,source=/home/whx/workspace/work/python_code/autotext/data/save_model/rnn,target=/models/rnn -e MOD
EL_NAME=rnn -t tensorflow/serving
```


### serving rest api
```
curl -d '{"signature_name": "call", "inputs":{"inputs":[[ 23,  22, 366,  65, 291,366, 65, 357, 349, 261]], "from_logits":true}}' -X POST  "http://127.0.0.1:8501/v1/models/rnn:predict"
curl -d '{"signature_name": "predict", "inputs":{"worlds":[[ 23,  22, 366,  65, 291,366, 65, 357, 349, 261]], "temperature":1.0}}' -X POST  "http://127.0.0.1:8501/v1/models/rnn:predict"
```


