# DP-FedAvg-GAN
DP-FedAvg-GAN is a GAN model adapted for tabular data based on the algorithm proposed in [Generative Models for Effective ML on Private, Decentralized Datasets](https://arxiv.org/abs/1911.06679).


## Usage

Change `DATASET_NAME` in line 2 of `constants.py` to select a certain dataset. `adult`, `clinical`, `covtype`, `credit` and `intrusion` are supported by default. Toggle `auto_split` in line 17 of `dp_fedavg_gan.py` to switch between an even split (True) or a biased split (False).

Modify `HyperParam` in `util.py` to fine tune the training process. The parameters are quite self-documented.


## Output
Synthesized table is saved as `syn_path`.

On each `save_step`, the state dicts of the generator and discriminators are saved in `out_folder`. These state dicts can be used by `syn_fake()` in `synthesizer.py` to generate fake table.

On each `eval_step` after `eval_start`, the table synthesized by the current generator is evaluated using metrics proposed in [SDGym](https://github.com/sdv-dev/SDGym/tree/v0.2.2).

After training, the generator loss, average loss of client discriminators, (macro) F1 scores as well as mean absolute errors are plotted respectively.
