# GuGo

This the the repository for the AAAI 2020 paper "[***Solving Sequential Text Classification as Board-Game Playing***](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-QianC.2311.pdf)".

GuGo (**G**ame-based j**u**mp labelin**g** m**o**del) is a jump labeling approach that generalizes the traditional successive labeling by developing a correspondence between sequential text classification and board-game playing.

## Overview

- code/ 
  contains the source codes.
- data/ 
  contains some datasets used for evaluating.

### Reqirements:

* Python (≥3.0)
* PyTorch (≥1.0)
* [BERT-Base](https://github.com/google-research/bert): Please initialize a pretrained BERT model (self.bert in class TextEmbedding) to obtain BERT embeddings.
* Hyperparameters are in _public.py.

### Citation

When referencing, please cite this paper as:

```
@inproceedings{GuGo,
  title={Solving Sequential Text Classification as Board-Game Playing},
  author={Chen Qian and Fuli Feng and Lijie Wen and Zhenpeng Chen and Li Lin and Yanan Zheng and Tat-Seng Chua},
  booktitle={Proceedings of The 34th AAAI Conference on Artificial Intelligence (AAAI)},
  year={2020}
}
```
