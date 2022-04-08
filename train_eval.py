import importlib
import json
import os
import time
from pathlib import Path

import torch.optim as optim
import torch.quantization.fx.utils
from tensorboardX import SummaryWriter

from data.data_loader_multigraph import GMDataset, get_dataloader
from eval import eval_model
from models.lossfunction import HammingLoss, PermutationLoss
from utils.dup_stdout_manager import DupStdoutFileManager
from utils.evaluation_metric import *
from utils.utils import update_params_from_cmdline


def init_config():
    from utils.config import config

    updated_cfg = update_params_from_cmdline(default_params=config)
    os.makedirs(updated_cfg.result_dir, exist_ok=True)
    with open(os.path.join(updated_cfg.result_dir, "settings.json"), "w") as f:
        json.dump(updated_cfg, f)

    return updated_cfg


def eval_one_epoch(model, dataloader):
    accs, f1_scores = eval_model(model, dataloader["test"])
    acc_dict = {
        "acc_{}".format(cls): single_acc for cls, single_acc in zip(dataloader["train"].dataset.classes, accs)
    }
    f1_dict = {
        "f1_{}".format(cls): single_f1_score
        for cls, single_f1_score in zip(dataloader["train"].dataset.classes, f1_scores)
    }
    acc_dict.update(f1_dict)
    acc_dict["matching_accuracy"] = torch.mean(accs)
    acc_dict["f1_score"] = torch.mean(f1_scores)

    return acc_dict


def train_eval_model(model, criterion, optimizer, dataloader, num_epochs, writer, resume=False, start_epoch=0):
    print("Start training..")

    since = time.time()
    dataloader["train"].dataset.set_num_graphs(config.num_graphs_in_matching_instance)
    dataset_size = len(dataloader["train"].dataset)

    device = next(model.parameters()).device
    print("model on device: {}".format(device))

    checkpoint_path = Path(config.result_dir) / "params"
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:
        params_path = os.path.join(config.warmstart_path, f"params.pt")
        print("Loading model parameters from {}".format(params_path))
        model.load_state_dict(torch.load(params_path))

        optim_path = os.path.join(config.warmstart_path, f"optim.pt")
        print("Loading optimizer state from {}".format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    # Evaluation only
    if config.evaluate_only:
        assert resume
        print(f"Evaluating without training..")
        acc_dict = eval_one_epoch(model, dataloader)

        time_elapsed = time.time() - since
        print(
            "Evaluation complete in {:.0f}h {:.0f}m {:.0f}s".format(
                time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60
            )
        )
        return model, acc_dict

    lr_params = config.TRAIN.lr_schedule
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_params["lr_milestones"], gamma=lr_params["lr_decay"]
    )

    acc_dict = dict()

    for epoch in range(start_epoch, num_epochs):
        print("-" * 10)
        print("Epoch {}/{}".format(epoch, num_epochs - 1))

        model.train()  # Set model to training mode

        print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_acc = 0.0
        epoch_acc = 0.0
        running_f1 = 0.0
        epoch_f1 = 0.0
        running_since = time.time()
        iter_num = 0
        statistic_step = config.STATISTIC_STEP

        # Iterate over data.
        for inputs in dataloader["train"]:
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt_list = [_.cuda() for _ in inputs["ns"]]
            graphs_list = [_.to("cuda") for _ in inputs["graphs"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                output = model(data_list, points_gt_list, graphs_list, n_points_gt_list, perm_mat_list)

                if config.TRAIN.LOSS_FUNC == "HammingLoss":
                    # computed model loss
                    loss = sum([criterion(s_pred, perm_mat) for s_pred, perm_mat in zip(output, perm_mat_list)])
                    loss /= len(output)

                    # computed model accuracy and f1 scores
                    tp, fp, fn = get_pos_neg_from_lists(output, perm_mat_list)
                    f1 = f1_score(tp, fp, fn)
                    acc, _, __ = matching_accuracy_from_lists(output, perm_mat_list)
                elif config.TRAIN.LOSS_FUNC == "PermutationLoss":
                    relax_sol, perm_sol = output["relax_sol"], output["perm_sol"]
                    loss = criterion(relax_sol, perm_mat_list[0], n_points_gt_list[0], n_points_gt_list[1])

                    tp, fp, fn = get_pos_neg(perm_sol, perm_mat_list[0])
                    f1 = f1_score(tp, fp, fn)
                    acc, _, __ = matching_accuracy(perm_sol, perm_mat_list[0])
                else:
                    raise ValueError("Unknown loss function type.")

                # backward + optimize
                loss.backward()
                optimizer.step()

                # statistics
                bs = perm_mat_list[0].size(0)
                running_loss += loss.item() * bs  # multiply with batch size
                epoch_loss += loss.item() * bs
                running_acc += acc.item() * bs
                epoch_acc += acc.item() * bs
                running_f1 += f1.item() * bs
                epoch_f1 += f1.item() * bs

                if iter_num % statistic_step == 0:
                    running_speed = statistic_step * bs / (time.time() - running_since)
                    loss_avg = running_loss / statistic_step / bs
                    acc_avg = running_acc / statistic_step / bs
                    f1_avg = running_f1 / statistic_step / bs
                    print(
                        "Epoch {:<4} Iter {:<4} {:>4.2f}sample/s Loss={:<8.4f} Accuracy={:<2.3} F1={:<2.3}".format(
                            epoch, iter_num, running_speed, loss_avg, acc_avg, f1_avg
                        )
                    )

                    running_acc = 0.0
                    running_f1 = 0.0
                    running_loss = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / dataset_size
        epoch_acc = epoch_acc / dataset_size
        epoch_f1 = epoch_f1 / dataset_size

        writer.add_scalars("Train", {"epoch_loss": epoch_loss,
                                     "epoch_acc": epoch_acc, }, epoch)

        if config.save_checkpoint:
            base_path = Path(checkpoint_path / "{:04}".format(epoch + 1))
            Path(base_path).mkdir(parents=True, exist_ok=True)
            path = str(base_path / "params.pt")
            torch.save(model.state_dict(), path)
            torch.save(optimizer.state_dict(), str(base_path / "optim.pt"))

        print(
            "Over whole epoch {:<4} -------- Loss: {:.4f} Accuracy: {:.3f} F1: {:.3f}".format(
                epoch, epoch_loss, epoch_acc, epoch_f1
            )
        )
        print()

        # Eval in each epoch
        acc_dict = eval_one_epoch(model, dataloader)
        writer.add_scalars("Eval", {"eval_acc": acc_dict["matching_accuracy"]}, epoch)

        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60
        )
    )

    return model, acc_dict


if __name__ == "__main__":
    config = init_config()

    # construct dataset and dataloader
    torch.manual_seed(config.RANDOM_SEED)
    dataset_len = {"train": config.TRAIN.EPOCH_ITER * config.TRAIN.BATCH_SIZE, "test": config.EVAL.SAMPLES}

    image_dataset = {
        x: GMDataset(config.DATASET_SETTING.DATASET_NAME, sets=x, length=dataset_len[x], obj_resize=(256, 256))
        for x in ("train", "test")
    }

    dataloader = {x: get_dataloader(image_dataset[x], config.TRAIN.BATCH_SIZE, fix_seed=(x == "test"), shuffle=True)
                  for x in ("train", "test")}

    # setting model
    module = importlib.import_module(config.TRAIN.MODULE)
    Net = module.Net
    model = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # setting loss function
    if config.TRAIN.LOSS_FUNC == "HammingLoss":
        criterion = HammingLoss()
    elif config.TRAIN.LOSS_FUNC == "PermutationLoss":
        criterion = PermutationLoss()
    else:
        raise ValueError("Unknown loss function type.")

    backbone_params = list(model.node_layers.parameters()) + list(model.edge_layers.parameters())
    backbone_params += list(model.final_layers.parameters())

    backbone_ids = [id(item) for item in backbone_params]

    new_params = [param for param in model.parameters() if id(param) not in backbone_ids]
    opt_params = [
        dict(params=backbone_params, lr=config.TRAIN.LR * 0.01),
        dict(params=new_params, lr=config.TRAIN.LR),
    ]
    optimizer = optim.Adam(opt_params)

    if not Path(config.result_dir).exists():
        Path(config.result_dir).mkdir(parents=True)

    writer = SummaryWriter(str(Path(config.result_dir) / "runs"))

    num_epochs = config.TRAIN.lr_schedule.num_epochs
    with DupStdoutFileManager(str(Path(config.result_dir) / "train_log.log")) as _:
        model, accs = train_eval_model(
            model,
            criterion,
            optimizer,
            dataloader,
            num_epochs=num_epochs,
            writer=writer,
            resume=config.warmstart_path is not None,
            start_epoch=0,
        )
