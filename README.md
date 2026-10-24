# IDML — Metric Learning Experiments

Research adaptation of [Introspective Deep Metric Learning](https://github.com/wzzheng/IDML). The upstream project remains the source of the method; this repository contains a CT folder adapter, ResNet/KAN experiments, and a local copy of `pytorch_metric_learning`.

## Repository map

| Path | Purpose |
| --- | --- |
| `train.py` | CUDA training, mixup, proxy-anchor loss, checkpoint export |
| `evaluate.py` | Evaluate a compatible saved checkpoint |
| `dataset/` | CT, CUB, Cars196, and Stanford Online Products loaders |
| `net/resnet.py`, `net/kan.py` | ResNet embedding/uncertainty networks and KAN layers |
| `losses.py`, `utils.py` | Objectives, nearest-neighbor retrieval, and evaluation helpers |
| `pytorch_metric_learning/` | Vendored upstream metric-learning utilities |

## Environment

The implementation uses CUDA operations directly; `--gpu-id -1` selects its historical multi-GPU path, not CPU inference. Prepare a CUDA-capable PyTorch installation with the matching torchvision version for your machine, then install the additional imports:

```bash
git clone https://github.com/Shayne-Pro/IDML.git
cd IDML
python -m pip install numpy scipy scikit-learn faiss-cpu pillow tqdm wandb
```

The environment is not locked. Full GPU training and metric reproduction require a compatible runtime, authorized data, and model weights; they have not been rerun as part of the documentation update. Backbone initialization may download torchvision pretrained weights.

## CT layout and labels

```text
data/CT/
  COVID_tr/       # training, label 0
  nonCOVID_tr/    # training, label 1
  COVID_ts/       # evaluation, label 0
  nonCOVID_ts/    # evaluation, label 1
```

The CT adapter selects folders by name and preserves the same two labels across splits. Lexicographic ImageFolder indices alone do not encode the train/evaluation partition. Both classes must be present in each selected split. The historical `dataset/ct_split.py` performs an unseeded image-level split; prepare patient-separated partitions yourself when images from a patient are correlated.

Other adapters expect `CUB_200_2011/`, `cars196/`, or `Stanford_Online_Products/` under the supplied data root. Consult their loader files for annotation filenames and obtain data under the original dataset terms.

## Training and evaluation

```bash
WANDB_MODE=disabled python train.py --dataset CT --model resnet50 --data-root ./data --LOG_DIR ./logs
python evaluate.py --dataset CT --model resnet50 --data-root ./data --resume /path/to/CT_resnet50_best.pth
```

Use the same embedding size, architecture, normalization setting, and dataset protocol for training and evaluation. The supported CLI backbones are `resnet18`, `resnet50`, and `resnet101`; missing Inshop/GoogLeNet/BN-Inception modules from older upstream versions are not advertised as available here. `--data-root` is resolved explicitly without changing the process working directory.

Training exports `model_state_dict` checkpoints named `DATASET_MODEL_best.pth` inside its generated log directory. This repository does not provide a trained CT checkpoint or a verified CT performance claim.

## Provenance

Keep upstream and vendored-code notices when redistributing. This repository currently provides no new repository-level license grant; do not assume the absence of a GitHub fork badge establishes original authorship.

## Regression checks

```bash
python -m unittest discover -s tests -v
```

These checks use tiny generated PNGs to verify folder partitioning and class labels, without model weights or CUDA.
