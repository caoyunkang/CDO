import numpy as np
from torch.utils.data import Sampler
from ..ad_dataset import DefaultAD


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


class BalancedBatchSampler(Sampler):
    def __init__(self,
                 batch_size,
                 dataset: DefaultAD):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset

        self.normal_generator = self.random_generator(self.dataset.normal_idx)
        self.outlier_generator = self.random_generator(self.dataset.outlier_idx)
        if len(self.dataset.outlier_idx) != 0:
            self.n_normal = self.batch_size // 2
            self.n_outlier = self.batch_size - self.n_normal
        else:
            self.n_normal = self.batch_size
            self.n_outlier = 0

        self.length = (len(self.dataset.normal_idx) // self.n_normal) + 1

    @staticmethod
    def random_generator(idx_list):
        while True:
            if len(idx_list) == 0:  # 防止空列表
                break
            random_list = np.random.permutation(idx_list)
            for i in random_list:
                yield i

    def __len__(self):
        return self.length

    # def __iter__(self):
    #     for _ in range(self.length):
    #         for idx in self.get_next_batch():  # 遍历批次中的每个索引
    #             yield idx
    #
    # def get_next_batch(self):
    #     batch = []
    #     for _ in range(self.n_normal):
    #         batch.append(next(self.normal_generator))
    #     for _ in range(self.n_outlier):
    #         try:
    #             batch.append(next(self.outlier_generator))
    #         except StopIteration:
    #             self.outlier_generator = self.random_generator(self.dataset.outlier_idx)
    #             batch.append(next(self.outlier_generator))
    #     print(batch)
    #     return batch

    def __iter__(self):
        for _ in range(self.length):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_outlier):
                try:
                    batch.append(next(self.outlier_generator))
                except StopIteration:
                    self.outlier_generator = self.random_generator(self.dataset.outlier_idx)
                    batch.append(next(self.outlier_generator))

            # print(f"Batch {batch}")

            yield batch
