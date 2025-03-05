import os
import pandas as pd

def read_csv(root_dir, sub_dir, dataset_list, name_formatting:str, required_metrics, return_average:bool):
    def _compute_average(df: pd.DataFrame, ):
        filtered_df = df[(df.index != 'Average') & (df.sum(axis=1) != 0)]
        avg_values = filtered_df.mean(axis=0)
        return avg_values

    path_list = []

    ######## dataset, category name
    for dataset_name in dataset_list:
        full_path = os.path.join(root_dir, sub_dir[0], name_formatting.format(sub_dir[1], dataset_name))
        path_list.append(full_path)

    if return_average:
        df_list = []
        index_list = []
        for path, dataset_name in zip(path_list, dataset_list):
            try:
                df = pd.read_csv(path, index_col=0)
                avg_values = _compute_average(df)
                df_list.append(avg_values)
                index_list.append(dataset_name)
            except Exception as e:
                print(f'Error in reading {path} for {dataset_name}: {str(e)}')
                return None
        result_df = pd.DataFrame(df_list, index=index_list)
    else:
        df_list = []
        for path, dataset_name in zip(path_list, dataset_list):
            try:
                df = pd.read_csv(path, index_col=0)
                df_list.append(df)
            except Exception as e:
                print(f'Error in reading {path} for {dataset_name}: {str(e)}')
                return None
        result_df = pd.concat(df_list, axis=0)
    result_df = result_df.loc[:, required_metrics]

    return result_df

def aggregate_csv(dfs, metrics, save_root, save_name):
    # dfs = {'A': A, 'B': B, 'C': C}
    # 创建一个字典用于保存拼接结果
    combined_dfs = {}

    # 对每个指标进行处理
    for metric in metrics:
        # 创建一个空的列表来保存每个 DataFrame 对应指标的列
        combined_data = []

        # 将每个 DataFrame 对应指标的列拼接在一起
        for name, df in dfs.items():
            combined_data.append(df[metric].rename(f"{name}"))

        # 将拼接后的数据转化为一个新的 DataFrame
        combined_df = pd.concat(combined_data, axis=1)

        # 将该指标的 DataFrame 保存到字典中
        combined_dfs[metric] = combined_df

    # 保存每个指标对应的 DataFrame 到 CSV 文件
    for metric, df in combined_dfs.items():
        save_path = os.path.join(save_root, f"{save_name}_{metric}.csv")
        df.to_csv(save_path, float_format='%.2f')
        print(f"Saved {save_path}")
