![LOGO](asset/logo.png)  

Reproducible material for **Stable neural network-based traveltime tomography using hard-constrained measurements - Taufik M., Alkhalifah T., and Waheed U. B.**


# Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: folder containing logo.
* :open_file_folder: **data**: folder containing the cropped [Marmousi](https://wiki.seg.org/wiki/Dictionary:Marmousi_model) model.
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details).
* :open_file_folder: **scripts**: set of (bash and python) scripts used to run multiple experiments.
* :open_file_folder: **src**: folder containing materials for the *hcpinnseikonal* package.

## Notebooks
The following notebooks are provided:

- :orange_book: ``Example-1-marmousi-2d.ipynb``: notebook performing a surface tomography acquisition with sparse source-receiver sampling using the cropped 2D Marmousi model.

## Scripts
The following scripts are provided:

- :page_with_curl: ``Example-1.sh``: script to perform different receiver sampling experiment for the acquisition perform in the ``Example-1.ipynb`` and ``Example-1.py``.

## Getting started
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate hcpinnseikonal
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz equipped with a single NVIDIA Quadro RTX 8000 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite us 
If you find our work useful in your research, please cite:
```
@article{taufik2024stable,
  title={Stable neural network-based traveltime tomography using hard-constrained measurements},
  author={Taufik, Mohammad H and Alkhalifah, Tariq and Waheed, Umair bin},
  journal={Geophysics},
  volume={89},
  number={6},
  pages={U87--U99},
  year={2024},
  publisher={Society of Exploration Geophysicists}
}
```
