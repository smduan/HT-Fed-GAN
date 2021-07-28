import os
import warnings
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

import statistic_all as statistic_test
import utility_all as utility_test
from synthesizer import synthesize
from util import HyperParam


class RealtimeEvaluator:
    def __init__(self, *, opt, trans, gen):
        warnings.filterwarnings(action='ignore', category=FutureWarning)
        # warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

        self._opt: HyperParam = opt
        self._trans = trans
        self._gen = gen

        # loss
        self._d_loss, self._g_loss = [], []
        # utility
        self._f1, self._macro_f1, self._mae = [], [], []
        self._test, self._meta, _, _ = utility_test.load_dataset('{0}.json'.format(self._opt.dataset_name), benchmark=True)
        # statistic
        self._real_data = pd.read_csv(self._opt.train_path, dtype='str').apply(lambda x: x.str.strip(' \t.'))
        self._x_real = statistic_test.read_data(self._real_data)

    # stupid
    def set_gen(self, gen):
        self._gen = gen

    def collect_loss(self, d_loss: float, g_loss: float):
        self._d_loss.append(d_loss)
        self._g_loss.append(g_loss)

    def plot_loss(self):
        assert len(self._d_loss) == len(self._g_loss)
        xticks = np.arange(0, len(self._d_loss) + self._opt.save_step, self._opt.save_step)
        # yticks = np.arange(0, .65, .05)

        plt.figure(figsize=[20, 12])
        plt.plot(self._d_loss, '-+')
        plt.xticks(xticks)
        plt.legend(['loss d'])
        plt.savefig(os.path.join(self._opt.out_folder, 'loss_d.png'))
        plt.show()

        plt.figure(figsize=[20, 12])
        plt.plot(self._g_loss, '-+')
        plt.xticks(xticks)
        plt.legend(['loss g'])
        plt.savefig(os.path.join(self._opt.out_folder, 'loss_g.png'))
        plt.show()

    def analyse_ml_performance(self, epoch: int):
        if epoch < self._opt.eval_start or not (epoch % self._opt.eval_step == 0 or epoch == self._opt.n_epoch - 1):
            return
        sdict = {name: state.detach().clone() for name, state in self._gen.state_dict().items()}
        try:
            synthesized_data = synthesize(self._opt, self._opt.dataset_size, sdict, self._trans, None)
            synthesized_data = synthesized_data.astype(dtype='str', copy=False)
            synthesized_data = synthesized_data.apply(lambda x: x.str.strip(' \t.'))
            synthesized = utility_test.read_data(synthesized_data)
            # x_fake & synthesized are identical
            # x_fake = statistic_test.read_data(synthesized_data)
        except IndexError:
            synthesized = None
        if self._opt.dataset_name == 'adult':
            self._do_utility_test(synthesized, ('income_bracket', 'binary'), 'hours_per_week')
            self._do_statistic_test(synthesized, 'education')
        elif self._opt.dataset_name == 'clinical':
            self._do_utility_test(synthesized, ('DEATH_EVENT', 'binary'), 'age')
            self._do_statistic_test(synthesized, 'ejection_fraction')
        elif self._opt.dataset_name == 'covtype':
            self._do_utility_test(synthesized, ('label', 'multi'), 'Elevation')
            self._do_statistic_test(synthesized, 'Elevation')
        elif self._opt.dataset_name == 'credit':
            self._do_utility_test(synthesized, ('label', 'binary'), 'Amount')
            self._do_statistic_test(synthesized, 'Amount')
        elif self._opt.dataset_name == 'intrusion':
            self._do_utility_test(synthesized, ('label', 'multi'), 'count')
            self._do_statistic_test(synthesized, 'count')
        # else:
        #     raise ValueError(self._opt.dataset_name)

    def _do_utility_test(self, synthesized, cls_col_name: Tuple[str, str], reg_col_name: str):
        if cls_col_name[1] == 'binary':
            cls_critic = 'f1'
            cls_score_list = self._f1
            cls_evaluate_func = utility_test.Evaluate_binary_classification
        elif cls_col_name[1] == 'multi':
            cls_critic = 'macro_f1'
            cls_score_list = self._macro_f1
            cls_evaluate_func = utility_test._evaluate_multi_classification
        else:
            raise ValueError(f'err classification {cls_col_name[1]}')
        reg_critic = 'mae'
        reg_score_list = self._mae
        reg_evaluate_func = utility_test._evaluate_regression

        if synthesized is None:
            cls_score_list.append(np.full(4, fill_value=np.nan))
            reg_score_list.append(np.full(4, fill_value=np.nan))
            print('score: nan')
            return
        scores_cls = cls_evaluate_func(synthesized, self._test, self._meta, cls_col_name[0])
        print(scores_cls)
        scores_reg = reg_evaluate_func(synthesized, self._test, self._meta, reg_col_name)
        print(scores_reg)
        cls_score_list.append(scores_cls[cls_critic].values)
        # mae could run up to 1e10; should be below 1.5 on the given datasets
        reg_score_list.append(np.minimum(1.999, scores_reg[reg_critic].values))

    def _do_statistic_test(self, x_fake, col_name):
        if x_fake is None:
            return
        num_real = num_fake = len(self._real_data)
        col_loc = self._real_data.columns.get_loc(col_name)
        a, b = self._x_real[:, col_loc], x_fake[:, col_loc]
        statistic_test.cdf(a.reshape((num_real,)), b.reshape((num_fake,)), col_name, 'cumulative probability')

    def plot_ml_info(self):
        if len(self._f1):
            self._do_plot(
                data=self._f1,
                legend=['DecisionTreeClassifier', 'AdaBoostClassifier', 'LogisticRegression', 'MLPClassifier'],
                save_path='./saves/f1.png',
                ytick=np.arange(0, .65, .05)
            )
        if len(self._macro_f1):
            self._do_plot(
                self._macro_f1,
                ['DecisionTreeClassifier', 'MLPClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier'],
                './saves/macro_f1.png',
            )
        if len(self._mae):
            self._do_plot(
                self._mae,
                ['Lasso', 'MLPRegressor', 'GradientBoostingRegressor', 'RandomForestRegressor'],
                './saves/mae.png',
            )

    @staticmethod
    def _do_plot(data: list, legend: Iterable[str], save_path: str, ytick=None):
        data = np.array(data)  # [np.arange(0, self._opt.n_epoch, self._opt.save_per)]
        plt.figure(figsize=[20, 12])
        plt.plot(data, '-+', alpha=.7)
        # plot_step = 1
        # plt.xticks(np.arange(0, len(data) + plot_step, plot_step))
        if ytick is not None:
            plt.yticks(ytick)
        plt.legend(legend)
        plt.savefig(save_path)
        plt.show()
