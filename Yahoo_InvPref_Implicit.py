import json

import global_config
from dataloader import BaseImplicitBCELossDataLoader, YahooImplicitBCELossDataLoader
import torch
import numpy as np

from evaluate import ImplicitTestManager
from models import InvPrefImplicit
from train import ImplicitTrainManager
from utils import draw_score_pic, merge_dict, _show_me_a_list_func, draw_loss_pic_one_by_one, query_user, query_str, \
    mkdir, query_int, get_class_name_str, _mean_merge_dict_func, show_me_all_the_fucking_result

torch.cuda.set_device(0)
DEVICE: torch.device = torch.device('cuda')

MODEL_CONFIG: dict = {
    'env_num': 2,
    'factor_num': 40,
    'reg_only_embed': True,
    'reg_env_embed': False,
    'use_lgn': True,
    'graph_path': '/storage/pbwei/INV-LGN/data/yahoo_4/s_pre_adj_mat.npz'
}

TRAIN_CONFIG: dict = {
    "batch_size": 8192,
    "epochs": 1000,
    "cluster_interval": 5,
    "evaluate_interval": 10,
    "lr": 0.005,
    "invariant_coe": 3.351991776096847,
    "env_aware_coe": 9.988658447411407,
    "env_coe": 9.06447753571379,
    "L2_coe": 3.1351402017943117,
    "L1_coe": 0.4935216278026648,
    "alpha": 1.9053711444718746,
    "use_class_re_weight": True,
    "use_recommend_re_weight": False,
    "test_begin_epoch": 0,
    "begin_cluster_epoch": None,
    "stop_cluster_epoch": None
}

EVALUATE_CONFIG: dict = {
    'top_k_list': [3, 5, 7],
    'test_batch_size': 1024,
    'eval_k': 5,
    'eval_metric': 'ndcg'
}

RANDOM_SEED_LIST = [17373331, 17373511, 17373423]

DATASET_PATH = '/Yahoo_all_data'
METRIC_LIST = ['ndcg', 'recall', 'precision']


def main(
        device: torch.device,
        model_config: dict,
        train_config: dict,
        evaluate_config: dict,
        data_loader: BaseImplicitBCELossDataLoader,
        random_seed: int,
        silent: bool = False,
        auto: bool = False,
        query: bool = True,
):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    model: InvPrefImplicit = InvPrefImplicit(
        user_num=data_loader.user_num,
        item_num=data_loader.item_num,
        env_num=model_config['env_num'],
        factor_num=model_config['factor_num'],
        reg_only_embed=model_config['reg_only_embed'],
        reg_env_embed=model_config['reg_env_embed'],
        use_lgn = model_config['use_lgn'],
        graph_path = model_config['graph_path'],
        device = device
    )
    model = model.to(device)

    evaluator: ImplicitTestManager = ImplicitTestManager(
        model=model,
        data_loader=data_loader,
        test_batch_size=evaluate_config['test_batch_size'],
        top_k_list=evaluate_config['top_k_list'],
        use_item_pool=True
    )

    train_tensor: torch.LongTensor = torch.LongTensor(data_loader.train_data_np).to(device)

    print(train_tensor.shape)
    assert train_tensor.shape[1] == 3

    train_manager: ImplicitTrainManager = ImplicitTrainManager(
        model=model, evaluator=evaluator, training_data=train_tensor,
        device=device,
        batch_size=train_config['batch_size'], epochs=train_config['epochs'],
        cluster_interval=train_config['cluster_interval'],
        evaluate_interval=train_config['evaluate_interval'],
        lr=train_config['lr'], invariant_coe=train_config['invariant_coe'],
        env_aware_coe=train_config['env_aware_coe'], env_coe=train_config['env_coe'],
        L2_coe=train_config['L2_coe'], L1_coe=train_config['L1_coe'], alpha=train_config['alpha'],
        use_class_re_weight=train_config['use_class_re_weight'],
        test_begin_epoch=train_config['test_begin_epoch'], begin_cluster_epoch=train_config['begin_cluster_epoch'],
        stop_cluster_epoch=train_config['stop_cluster_epoch'],
        use_recommend_re_weight=train_config['use_recommend_re_weight']
    )

    train_tuple, test_tuple, cluster_tuple = train_manager.train(silent=silent, auto=auto)

    test_result_list = test_tuple[0]
    result_merged_by_metric: dict = merge_dict(test_result_list, _show_me_a_list_func)

    eval_k: int = evaluate_config['eval_k']
    eval_metric: str = evaluate_config['eval_metric']

    stand_result: np.array = np.array(merge_dict(result_merged_by_metric[eval_metric], _show_me_a_list_func)[eval_k])

    best_performance = np.max(stand_result)
    best_indexes: list = np.where(stand_result == best_performance)[0].tolist()
    fucking_result: dict = show_me_all_the_fucking_result(
        result_merged_by_metric,
        METRIC_LIST,
        evaluate_config['top_k_list'],
        best_indexes[0]
    )

    if not auto:
        print('Best {}@{}:'.format(eval_metric, eval_k), best_performance, best_indexes)

    if not silent and not auto:
        loss_dict_list: list = train_tuple[0]

        loss_dict: dict = merge_dict(loss_dict_list, _show_me_a_list_func)
        draw_loss_pic_one_by_one(len(train_tuple[1]), **loss_dict)

        # all_test_result: dict = {}

        for metric in result_merged_by_metric.keys():
            result_merged_by_k = merge_dict(result_merged_by_metric[metric], _show_me_a_list_func)
            # all_test_result[metric] = merge_dict(result_merged_by_metric[metric], _show_me_a_list_func)
            dict_to_show: dict = dict()
            for key in result_merged_by_k.keys():
                dict_to_show[str(key)] = result_merged_by_k[key]
            draw_score_pic(test_tuple[1], **dict_to_show)

        draw_score_pic(cluster_tuple[2], diff_num=cluster_tuple[0])

        dict_to_show: dict = dict()
        temp_dict = merge_dict(cluster_tuple[1], _show_me_a_list_func)
        for key in temp_dict.keys():
            dict_to_show[str(key)] = temp_dict[key]
        draw_score_pic(cluster_tuple[2], **dict_to_show)

    result_json: str = json.dumps(result_merged_by_metric, indent=4)
    evaluate_config_json: str = json.dumps(evaluate_config, indent=4)
    model_config_json: str = json.dumps(model_config, indent=4)
    train_config_json: str = json.dumps(train_config, indent=4)

    if query:
        would_save: bool = query_user('Would you like to save the result?')

        if would_save and not auto:
            result_name: str = query_str('The name of result?').replace('\t ', '')

            result_path: str = global_config.RESULT_SAVE_PATH + '/' + get_class_name_str(model) + '/' + result_name

            describe_info: str = query_str('describe?')

            mkdir(result_path)

            with open(result_path + '/result.txt', 'w') as output:
                output.write('best {}@{}: {}, {}\n'.format(eval_metric, eval_k, best_performance, best_indexes))
                output.write(result_json)
                output.close()

            with open(result_path + '/config.txt', 'w') as output:
                output.write('rand seed: '+ str(random_seed) + '\n')
                output.write(model_config_json + '\n')
                output.write(train_config_json + '\n')
                output.write(evaluate_config_json + '\n')
                output.write(str(list(model.modules())) + '\n')
                output.close()

            with open(result_path + '/describe.txt', 'w') as output:
                output.write(describe_info + '\n')
                output.close()

        while True and not auto:
            command: int = query_int('Would you like to\n0. exit\n1. view result\n2. view config', {0, 1, 2})
            if command == 0:
                break
            elif command == 1:
                print(result_json)
            else:
                print(evaluate_config)
                print(train_config_json)
                print(model_config_json)

    return best_performance, best_indexes, fucking_result


if __name__ == '__main__':
    loader: YahooImplicitBCELossDataLoader = YahooImplicitBCELossDataLoader(
        dataset_path=global_config.DATASET_PATH + DATASET_PATH,
        device=DEVICE, has_item_pool_file=True
    )

    best_metric_perform_list: list = []
    all_metric_results_list: list = []

    for seed in RANDOM_SEED_LIST:
        print()
        print('Begin seed:', seed)
        best_perform, _, all_metric_result = main(
            device=DEVICE, model_config=MODEL_CONFIG, train_config=TRAIN_CONFIG,
            evaluate_config=EVALUATE_CONFIG, data_loader=loader, random_seed=seed, query=False
        )

        best_metric_perform_list.append(best_perform)
        all_metric_results_list.append(all_metric_result)

    merged_all_metric: dict = merge_dict(all_metric_results_list, _mean_merge_dict_func)

    merged_all_metric_str: str = json.dumps(merged_all_metric, indent=4)
    print(merged_all_metric_str)

    print('Best perform mean:', np.mean(best_metric_perform_list))
    print('Best perform var:', np.var(best_metric_perform_list))
    print('Best perform std:', np.std(best_metric_perform_list))
    print('Random seed list:', RANDOM_SEED_LIST)
