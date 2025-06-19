from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *


def main():
    
    datafolder = '/home/ec2-user/ecg_ptbxl_benchmarking/data/ptbxl/'
    outputfolder = '../output/'

    models = [
        conf_fastai_xresnet1d101,
        # conf_fastai_resnet1d_wang,
        # conf_fastai_lstm,
        # conf_fastai_lstm_bidir,
        # conf_fastai_fcn_wang,
        # conf_fastai_inception1d,
        # conf_wavelet_standard_nn,
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        # ('exp0', 'all'),
        ('exp1', 'diagnostic'),
        # ('exp1.1', 'subdiagnostic'),
        # ('exp1.1.1', 'superdiagnostic'),
        # ('exp2', 'form'),
        # ('exp3', 'rhythm')
       ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        e.perform()
        e.evaluate()

    # generate great summary table
    utils.generate_ptbxl_summary_table()


if __name__ == "__main__":
    main()
