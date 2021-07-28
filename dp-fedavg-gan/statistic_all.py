import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import CATEGORICAL, COL_TYPE_ALL, CONTINUOUS, DATASET_NAME


def project_table(data, meta):
    values = np.zeros(shape=data.shape, dtype='float32')
    for id_, info in enumerate(meta):
        if info['type'] == CONTINUOUS:
            values[:, id_] = data.iloc[:, id_].values.astype('float32')
        else:
            mapper = dict([(item, id) for id, item in enumerate(info['i2s'])])
            mapped = data.iloc[:, id_].apply(lambda x: mapper[x]).values
            values[:, id_] = mapped
    return values


def read_data(df):
    col_type = COL_TYPE_ALL[DATASET_NAME]

    meta = []
    for id_, info in enumerate(col_type):
        if info[1] == CONTINUOUS:
            meta.append({
                "name": info[0],
                "type": info[1],
                "min": np.min(df.iloc[:, id_].values.astype('float')),
                "max": np.max(df.iloc[:, id_].values.astype('float'))
            })
        else:
            if info[1] == CATEGORICAL:
                value_count = list(dict(df.iloc[:, id_].value_counts()).items())  # 同一列不同取值及其数量
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))  # 取值提出来
            else:
                mapper = info[2]

            meta.append({
                "name": info[0],
                "type": info[1],
                "size": len(mapper),
                "i2s": mapper
            })

    tdata = project_table(df, meta)
    return tdata


def cdf(data_r, data_f, xlabel, ylabel):
    #     功能：用于生成某项属性的密度分图
    #     参数：data_r 真实数据的某项属性 np数组
    #           data_f 生成数据的某项属性 np数组
    #         xlabel x轴名称  ylabel y轴名称

    # if not os.path.exists(save_dir + '/cdf'):
    #    os.makedirs(save_dir + '/cdf')

    # axis_font = {'fontname': 'Arial', 'size': '18'}

    # Cumulative Distribution
    # scaler = MinMaxScaler()  # 数据归一化
    # real = scaler.fit_transform(data_r)
    # fake = scaler.fit_transform(data_f)
    # x1 = np.sort(real)
    # x2 = np.sort(fake)

    x1 = np.sort(data_r)
    x2 = np.sort(data_f)
    y1 = np.arange(1, len(data_r) + 1) / len(data_r)  # 计算出现概率
    y2 = np.arange(1, len(data_f) + 1) / len(data_f)

    fig = plt.figure()

    plt.xlabel(xlabel)  # , **axis_font)
    plt.ylabel(ylabel)  # , **axis_font)

    plt.grid()
    plt.margins(0.02)

    plt.plot(x1, y1, marker='o', linestyle='none', label='Real Data')  # , ms=8)
    plt.plot(x2, y2, marker='o', linestyle='none', label='Fake Data')  # , alpha=0.5, ms=5)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

    plt.show()
    # plt.savefig(xlabel)

    plt.close(fig)


if __name__ == '__main__':
    # 加载数据

    real_data = pd.read_csv("./{0}/{0}_train.csv".format(DATASET_NAME), dtype='str', header=0)  # 真实数据路径
    fake_data = pd.read_csv("./{0}_fake.csv".format(DATASET_NAME), dtype='str', header=0)  # 生成数据路径

    # 对数据进行处理
    x_real = real_data.apply(lambda x: x.str.strip(' \t.'))
    x_real = read_data(x_real)
    x_fake = fake_data.apply(lambda x: x.str.strip(' \t.'))
    x_fake = read_data(x_fake)

    # 获取数据行数
    num_real = len(real_data)
    num_fake = len(fake_data)

    # 选取需要查看分布的一列
    for col_loc in range(len(real_data.columns)):
        a = x_real[:, col_loc]  # 填写所需属性列数0~14
        b = x_fake[:, col_loc]
        col_name = real_data.columns[col_loc]
        cdf(a.reshape((num_real,)), b.reshape((num_fake,)), col_name, "Cumulative probability")  # 列变行

    # %%

    # 生成的数据以横坐标命名, 可以自己在cdf函数中添加保存路径参数
    #  作用在save部分
