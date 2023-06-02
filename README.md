# text_normalization

- More info in the model card: https://huggingface.co/maximxls/text-normalization-ru-terrible

### Training replication procedure
1. Clone this repo: `git clone https://github.com/maximxlss/text_normalization`
2. `cd text_normalization`
3. Install requirements: `pip install -r requirements.txt`
4. Install [PyTorch](https://pytorch.org/get-started/locally/)
5. Download `ru_train.csv` from [this Kaggle challenge](https://www.kaggle.com/c/text-normalization-challenge-russian-language)
6. Run `python preprocess.py` (takes time)
7. Run `python train_tokenizer.py` (also takes time)
8. Tweak settings in `train.py`
9. Run `python train.py`
10. I have reset the scheduler (see `train.py`) manually when training so keep that in mind. You can see the details of the training process in the [metrics](https://huggingface.co/maximxls/text-normalization-ru-terrible/tensorboard)
