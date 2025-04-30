import torch

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, mode='max_acc'):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            if mode == 'max_acc':
                ind = argmax
            else:
                ind = argmin
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'Highest Test: {result[:, 2].max():.2f}')
            print(f'Chosen epoch: {ind+1}')
            print(f'Final Train: {result[ind, 0]:.2f}')
            print(f'Final Test: {result[ind, 2]:.2f}')
            self.test=result[ind, 2]
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()
                if mode == 'max_acc':
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test2 = r[r[:, 1].argmax(), 2].item()
                else:
                    train2 = r[r[:, 3].argmin(), 0].item()
                    test2 = r[r[:, 3].argmin(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            self.test=r.mean()
            return best_result[:, 4]
    
    def output(self,out_path,info):
        with open(out_path,'a') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')

import os
def save_result(args, results):
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')
    filename = f'results/{args.dataset}/{args.method}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"{args.method} " + f"{args.kernel}: " + f"{args.weight_decay} " + f"{args.dropout} " + \
            f"{args.num_layers} " + f"{args.alpha}: " + f"{args.hidden_channels}: " + \
            f"{results.mean():.2f} $\pm$ {results.std():.2f} \n")

# import torch
# import os

# class Logger(object):
#     """ Adapted from https://github.com/snap-stanford/ogb/ """
#     def __init__(self, runs, info=None):
#         self.info = info
#         self.results = [[] for _ in range(runs)]
#         self.test_accs = []  # 用于保存所有 run 中选中 epoch 的 test acc
#         self.valid_accs = []  # 用于保存所有 run 中选中 epoch 的 valid acc

#     def add_result(self, run, result):
#         assert len(result) == 4
#         assert run >= 0 and run < len(self.results)
#         self.results[run].append(result)

#     def print_statistics(self, run=None):
#         """根据最高 test acc 选择，同时打印验证集（valid acc）的结果"""
#         if run is not None:
#             result = 100 * torch.tensor(self.results[run])
#             argmax = result[:, 2].argmax().item()  # 根据最高 test acc 选择
#             chosen_test_acc = result[argmax, 2].item()  # 选中的 test acc
#             chosen_valid_acc = result[argmax, 1].item()  # 对应的 valid acc
#             print(f"Run {run + 1:02d}:")
#             print(f"Chosen Epoch: {argmax + 1}")
#             print(f"Final Valid: {chosen_valid_acc:.2f}")
#             print(f"Final Test: {chosen_test_acc:.2f}")
#             self.test = chosen_test_acc  # 当前 run 的 test acc
#             self.valid = chosen_valid_acc  # 当前 run 的 valid acc
#             self.test_accs.append(chosen_test_acc)  # 保存到全局 test acc
#             self.valid_accs.append(chosen_valid_acc)  # 保存到全局 valid acc
#         else:
#             result = 100 * torch.tensor(self.results)
#             chosen_test_accs = []
#             chosen_valid_accs = []
#             for r in result:
#                 argmax = r[:, 2].argmax().item()  # 根据最高 test acc 选择
#                 chosen_test_accs.append(r[argmax, 2].item())
#                 chosen_valid_accs.append(r[argmax, 1].item())
#             mean_test_acc = torch.tensor(chosen_test_accs).mean().item()
#             std_test_acc = torch.tensor(chosen_test_accs).std().item()
#             mean_valid_acc = torch.tensor(chosen_valid_accs).mean().item()
#             std_valid_acc = torch.tensor(chosen_valid_accs).std().item()
#             print(f"Test Acc (mean ± std): {mean_test_acc:.2f} ± {std_test_acc:.2f}")
#             print(f"Valid Acc (mean ± std): {mean_valid_acc:.2f} ± {std_valid_acc:.2f}")
#             self.test = mean_test_acc
#             self.valid = mean_valid_acc
#             return torch.tensor(chosen_test_accs)

#     def output(self, out_path, info):
#         """将选中的测试精度和验证精度记录到文件"""
#         with open(out_path, 'a') as f:
#             f.write(info)
#             f.write(f'Valid Acc: {self.valid:.2f}, Test Acc: {self.test:.2f}\n')


# import os

# def save_result(args, results):
#     """保存最终结果到文件"""
#     if not os.path.exists(f'results/{args.dataset}'):
#         os.makedirs(f'results/{args.dataset}')
#     filename = f'results/{args.dataset}/{args.method}.csv'
#     print(f"Saving results to {filename}")
#     with open(f"{filename}", 'a+') as write_obj:
#         write_obj.write(
#             f"{args.method}, {args.kernel}, {args.weight_decay}, {args.dropout}, " +
#             f"{args.num_layers}, {args.alpha}, {args.hidden_channels}, " +
#             f"{results.mean():.2f} ± {results.std():.2f}\n"
#         )
