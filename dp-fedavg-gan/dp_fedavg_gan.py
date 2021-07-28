from dataset import split_dataset
from read_biased_split import read_biased_split
from realtime_evaluator import RealtimeEvaluator
from synthesizer import synthesize
from train import train
from transformer import fit_transform
from util import HyperParam


def main():
    opt = HyperParam()
    opt.start()
    print(opt)
    opt.set_seed()

    # todo: change here
    auto_split = False
    if auto_split:
        transformer, raw_dataset = fit_transform(opt, opt.train_path, opt.dataset_dcol)
        cli_dl = split_dataset(opt, raw_dataset, split_ratio=[.15, .30, .55])
        # cli_dl = split_dataset(opt, raw_dataset, split_ratio=[1.])
    else:
        transformer, _ = fit_transform(opt, opt.train_path, opt.dataset_dcol)
        col_cnt_switch = {
            'adult': ('education', 3),
            'clinical': ('age', 3),
            'covtype': ('Elevation', 3),
            'credit': ('Amount', 3),
            'intrusion': ('flag', 3)
        }
        cli_dl = read_biased_split(opt, transformer, *col_cnt_switch[opt.dataset_name])

    # evaluator = None
    evaluator = RealtimeEvaluator(opt=opt, trans=transformer, gen=None)

    generator = train(opt, cli_dl, evaluator=evaluator)

    synthesize(opt, opt.dataset_size, generator.state_dict(), transformer, opt.syn_path)


if __name__ == '__main__':
    main()
