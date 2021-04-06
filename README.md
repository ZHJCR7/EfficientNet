## Deepfake detection based on EfficientNet network and its added attention mechanism

### Usage

- ###### to train the model

```python
python efficientnet_train.py
```

- ###### to test the model

```python
python efficientnet_test.py
```

- ###### to generate some metics of the model

```python
python efficientnet_eval.py
```

### Dataset

```
FaceForensicspp
|--manipulated_sequences
	|-Deepfakes
		|-c0
		   |-test
		   |-train
		   |-validation
		|-c23
		|-c40
	|-Face2Face
	|-FaceSwap
|--original_sequences
	|-YouTube
		|-c0
		   |-test
		   |-train
		   |-validation
		|-c23
```

### Experiment

![Figure_1](G:\DeepFake\Github\EfficientNet\picture\Figure_1.png)

### Reference

1. https://github.com/polimi-ispl/icpr2020dfdc
2. https://github.com/HongguLiu/Deepfake-Detection
3. https://github.com/JStehouwer/FFD_CVPR2020