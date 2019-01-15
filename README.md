# CoDraw Models

This repository contains models for the Collaborative Drawing (CoDraw) task.

## Installation

Dependencies:
* Python 3.6 or later
* PyTorch 0.4.0
  * **IMPORTANT**: Our pre-trained models are **not** compatible with PyTorch 0.4.1. Please use version 0.4.0 exactly.
* `Pillow` 5.1.0 (or compatible)
* `editdistance` 0.4 (or compatible)

You will need to clone both this repository and the [CoDraw dataset repository](https://github.com/facebookresearch/CoDraw) into side-by-side folders. The README for the dataset repository contains additional instructions for downloading required dataset files.

The following relative paths should be reachable from within this repository:
```console
$ ls -d ../CoDraw/Pngs
../CoDraw/Pngs
$ ls -d ../CoDraw/dataset/CoDraw_1_0.json
../CoDraw/dataset/CoDraw_1_0.json
```

### Pre-trained models

Pre-trained models can be downloaded from this link: [models.tar.gz](https://github.com/facebookresearch/codraw-models/releases/download/models/models.tar.gz). The archive contains a `models/` folder that should be placed at the root of this repository.

## Usage

### Automated evaluation

After downloading our pre-trained models, run `python eval_automatic.py` to calculate the machine-machine and script-based scene similarity numbers that we report in our paper.

### Training new models

Before training any models, please make sure that the `models` and `rl_models` folders exist within this repo: `mkdir -p models rl_models`.

Each of the following commands trains a subset of the models we report in the paper:
* `python baseline1_train.py`
* `python baseline2_train.py`
* `python baseline3_train.py`
* `python baseline4_train.py`

Trained models are loaded by the function `load_baseline1` in `baseline1_models.py` and its counterparts in `baseline2_models.py`, `baseline3_models.py`, `baseline4_models.py`. Note that all of these functions use hard-coded paths that match our pre-trained model release; you will probably need to change these paths if you train your own models. Also note that the training process of some of the later models relies on the existence of earlier ones.

### Evaluating playing with humans

The transcripts for the human-machine evaluation reported in our paper are in the `transcripts-eval-v1.json` file in this repository. To compute the scene similarity scores we report in our paper, update the `TRANSCRIPTS_PATH` variable in `eval_transcripts.py` and then run `python eval_transcripts.py`.

## Reference

If you find this code useful in your research, we'd really appreciate it if you could cite the following paper:

```
@article{CoDraw,
author = {Kim, Jin-Hwa and Kitaev, Nikita and Chen, Xinlei and Rohrbach, Marcus and Tian, Yuandong and Batra, Dhruv and Parikh, Devi},
journal = {arXiv preprint arXiv:1712.05558},
title = {{CoDraw: Collaborative Drawing as a Testbed for Grounded Goal-driven Communication}},
url = {http://arxiv.org/abs/1712.05558},
year = {2019}
}
```

## License

This repository is licensed under Creative Commons Attribution-NonCommercial 4.0 International Public License, as found in the LICENSE file.
