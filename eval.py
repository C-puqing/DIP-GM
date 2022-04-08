import time
from pathlib import Path

import torch

from utils.config import config
from utils.evaluation_metric import matching_accuracy, f1_score, get_pos_neg


def eval_model(model, dataloader, eval_epoch=None, verbose=False):
    print("Start evaluation..")
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(config.result_dir) / "params" / "params_{:04}.pt".format(eval_epoch))
        print("Loading model parameters from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    ds.set_num_graphs(config.num_graphs_in_matching_instance)
    classes = ds.classes
    cls_cache = ds.cls

    accs = torch.zeros(len(classes), device=device)
    f1_scores = torch.zeros(len(classes), device=device)

    for i, cls in enumerate(classes):
        if verbose:
            print("Evaluating class {}: {}/{}".format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0
        statistic_step = config.STATISTIC_STEP

        ds.set_cls(cls)
        acc_match_num = torch.zeros(1, device=device)
        acc_total_num = torch.zeros(1, device=device)
        tp = torch.zeros(1, device=device)
        fp = torch.zeros(1, device=device)
        fn = torch.zeros(1, device=device)
        for k, inputs in enumerate(dataloader):
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt = [_.cuda() for _ in inputs["ns"]]
            graphs = [_.to("cuda") for _ in inputs["graphs"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            batch_num = data_list[0].size(0)

            iter_num = iter_num + 1

            visualize = k == 0 and config.visualize
            visualization_params = {**config.visualization_params,
                                    **dict(string_info=cls, true_matchings=perm_mat_list)}
            with torch.set_grad_enabled(False):
                s_pred_list = model(
                    data_list,
                    points_gt,
                    graphs,
                    n_points_gt,
                    perm_mat_list,
                    visualize_flag=visualize,
                    visualization_params=visualization_params,
                )

            _acc_match_num, _acc_total_num, _tp, _fp, _fn = 0, 0, 0, 0, 0
            perm_sol = s_pred_list['perm_sol']
            _, _acc_match_num, _acc_total_num = matching_accuracy(perm_sol, perm_mat_list[0])
            _tp, _fp, _fn = get_pos_neg(perm_sol, perm_mat_list[0])
            # if config.TRAIN.MODULE == "models.DIP.model":
            #     node_matching = s_pred_list[0][0]  # the 0th element of tuple is node matching matrix.
            #     _, _acc_match_num, _acc_total_num = matching_accuracy(node_matching, perm_mat_list[0])
            #     _tp, _fp, _fn = get_pos_neg(node_matching, perm_mat_list[0])
            # elif config.TRAIN.MODULE == "models.power_iteration.model":
            #     integer_sol = s_pred_list["perm_mat"]
            #     _, _acc_match_num, _acc_total_num = matching_accuracy(integer_sol, perm_mat_list[0])
            #     _tp, _fp, _fn = get_pos_neg(integer_sol, perm_mat_list[0])
            # elif config.TRAIN.MODULE == "models.BB_GM.model":
            #     _, _acc_match_num, _acc_total_num = matching_accuracy(s_pred_list[0], perm_mat_list[0])
            #     _tp, _fp, _fn = get_pos_neg(s_pred_list[0], perm_mat_list[0])

            acc_match_num += _acc_match_num
            acc_total_num += _acc_total_num
            tp += _tp
            fp += _fp
            fn += _fn

            if iter_num % statistic_step == 0 and verbose:
                running_speed = statistic_step * batch_num / (time.time() - running_since)
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()

        accs[i] = acc_match_num / acc_total_num
        f1_scores[i] = f1_score(tp, fp, fn)
        if verbose:
            print("Class {} acc = {:.4f} F1 = {:.4f}".format(cls, accs[i], f1_scores[i]))

    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print("Matching accuracy")
    for cls, single_acc, f1_sc in zip(classes, accs, f1_scores):
        print("{} = {:.4f}, {:.4f}".format(cls, single_acc, f1_sc))
    print("average = {:.4f}, {:.4f}".format(torch.mean(accs), torch.mean(f1_scores)))

    return accs, f1_scores
