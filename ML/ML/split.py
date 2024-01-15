import pandas as pd
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
def split(file_path):
    df = pd.read_csv(file_path)

    # 将日期列转换为日期时间格式
    df['date'] = pd.to_datetime(df['date'])

    # 提取月份
    df['month'] = df['date'].dt.month

    # 提取周末信息
    df['weekend'] = df['date'].dt.dayofweek  # 0表示工作日，1表示周末

    # 提取小时
    df['hour'] = df['date'].dt.hour

    # 对列进行调整
    df.drop(df.columns[0], axis=1, inplace=True)
    cols = df.columns.tolist()
    new_order = cols[-3:] + cols[:-3]
    df = df[new_order]

    df.to_csv(file_path, index=False)
    # normalized(file_path)

# def normlization(file_path):
#     # 读取CSV文件
#     df = pd.read_csv(file_path)

#     # 选择需要归一化的列
#     columns_to_normalize = ["month","weekend","hour","HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]  # 将 'column1', 'column2', 'column3' 替换为你需要归一化的列名

#     # 标准化（Z-score归一化）
#     scaler = StandardScaler()
#     df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

#     # 对列进行调整
#     df.drop(df.columns[0], axis=1, inplace=True)
#     cols = df.columns.tolist()
#     new_order = cols[-3:] + cols[:-3]
#     df = df[new_order]

#     # 保存修改后的数据为新的CSV文件
#     df.to_csv(file_path, index=False)

def show():
    # load dataset
    dataset = read_csv('ETT-small/output.csv', header=0)
    print(dataset)
    values = dataset.values
    # specify columns to plot
    groups = [3, 4, 5, 6, 7, 8, 9]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.savefig('savefig_example.png')

# split("./ETT-small/test_set.csv")



def normalized(file_path):
    # df是一个Pandas DataFrame，包含了想要正则化的数据
    df = pd.read_csv(file_path)

    # 保存原始数据的统计参数
    min_max_params = df.min(), df.max()
    
    # 将统计参数保存到文件
    with open('min_max_params.pickle', 'wb') as f:
        pickle.dump(min_max_params, f)

    # 最小-最大正则化
    def min_max_normalize(df, min_max_params):
        min_vals, max_vals = min_max_params
        return (df - min_vals) / (max_vals - min_vals)

    
    # 正则化数据
    normalized_min_max = min_max_normalize(df, min_max_params)

    # 读取统计参数
    with open('min_max_params.pickle', 'rb') as f:
        loaded_min_max_params = pickle.load(f)

    # 反正则化函数
    def min_max_denormalize(normalized_df, min_max_params):
        min_vals, max_vals = min_max_params
        return normalized_df * (max_vals - min_vals) + min_vals

    # 对正则化数据进行反正则化
    original_min_max = min_max_denormalize(normalized_min_max, loaded_min_max_params)
    print(loaded_min_max_params)
    # 检查反正则化后的数据是否与原始数据相同
    print(df.head())
    print(original_min_max.head())
    normalized_min_max.to_csv(file_path)

# split("./ETT-small/train_set.csv")
def relative_photo(file_path):
    
    import seaborn as sns
    df = pd.read_csv(file_path)
    
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(df.corr(), square=True, annot=True, ax=ax) # heat_map
    # sns_pp = sns.pairplot(df) # relative_map
    figure.savefig("heat_map.png")
# split('./ETT-small/full_set.csv')
relative_photo("./ETT-small/full_set.csv")
