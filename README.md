# finetune_test


## MODEL yolov7
- doamin : objection detection (object = parking lots)
- get model from : https://github.com/WongKinYiu/yolov7
- get dataset from : https://public.roboflow.com/object-detection/pklot/2
  - data shape : 640 x 640
  - data format : YOLO v7 PyTorch
  - data file name : PKLot.v2-640.yolov7pytorch.zip

```shell
# prepare python env
cd model_zoo/yolov7
python -m venv venv && source venv/bin/activate
pip install -U pip && pip install -r requirements.txt

# prepare dataset



watch -d nvidia-smi
```


## MODEL llm (Using LLamaFactory)
- doamin : llm
- get model from : ANY of llama factory (huggingface)
- get dataset from : ANY of llama factory (huggingface)

```shell
# prepare python env
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
python -m venv llamafactory && source llamafactory/bin/activate
pip install -U pip
pip install -e ".[torch,metrics]"

# excute gui
llamafactory-cli webui
# http://0.0.0.0:7860/

watch -d nvidia-smi
```