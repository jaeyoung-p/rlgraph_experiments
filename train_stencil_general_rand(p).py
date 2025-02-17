import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
import os
import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.loader
from scripts.setup_system import setup_system
from scripts.setup_graph import setup_graph
from scripts.setup_simulator import setup_simulator
from task4feedback.fastsim.interface import (
    PythonMapper,
    Action,
    Simulator,
    SimulatorHandler,
)
from scripts.run_graph import run
from task4feedback.fastsim.models import TaskAssignmentNetDeviceOnly
import logging
from task4feedback.types import (
    DataID,
    DataInfo,
    TaskID,
    TaskInfo,
    TaskPlacementInfo,
    TaskRuntimeInfo,
    Device,
    Architecture,
)
from task4feedback.graphs import *
import wandb
from datetime import datetime

logging.disable(logging.CRITICAL)

LEVELS = -1


@dataclass
class Args:
    hidden_dim = 64
    seed: int = 1
    torch_deterministic: bool = True
    env_id: str = "stencil"
    learning_rate: float = 2.5e-4
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_iterations: int = 25000
    graphs_per_update: int = 24
    reward: str = "percent_improvement"
    load_model: bool = False


def init_weights(m):
    if isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)


def logits_to_actions(
    logits, action=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.distributions.Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action), probs.entropy()


class GreedyNetworkMapper(PythonMapper):
    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator):
        data = simulator.observer.local_graph_features(candidates, k_hop=1)
        with torch.no_grad():
            d, v = self.model.forward(data)
            # Choose argmax of network output for priority and device assignment
            dev_per_task = torch.argmax(d, dim=-1)
            action_list = []
            for i in range(len(candidates)):
                # Check if p_per_task and dev_per_task are scalars
                if dev_per_task.dim() == 0:
                    dev_task = dev_per_task.item()
                else:
                    dev_task = dev_per_task[i].item()
                a = Action(
                    candidates[i],
                    i,
                    dev_task,
                    0,
                    0,
                )
                action_list.append(a)
        return action_list


class RandomNetworkMapper(PythonMapper):

    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator, output=None):
        data = simulator.observer.local_graph_features(candidates, k_hop=1)

        with torch.no_grad():
            self.model.eval()
            d, v = self.model.forward(data)
            self.model.train()

            # sample from network output
            dev_per_task, dlogprob, _ = logits_to_actions(d)

            if output is not None:
                output["candidates"] = candidates
                output["state"] = data
                output["dlogprob"] = dlogprob
                output["value"] = v
                output["dactions"] = dev_per_task

            action_list = []
            for i in range(len(candidates)):

                a = Action(
                    candidates[i],
                    i,
                    dev_per_task,
                    0,
                    0,
                )

                action_list.append(a)
        return action_list

    def evaluate(self, obs, daction, paction):
        p, d, v = self.model.forward(obs)
        _, plogprob, pentropy = logits_to_actions(p, paction)
        _, dlogprob, dentropy = logits_to_actions(d, daction)
        return (p, plogprob, pentropy), (d, dlogprob, dentropy), v


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def my_app(cfg: DictConfig) -> None:
    args = Args()
    run_name = (
        f"ppo_{args.env_id}_{cfg.dag.stencil.width}x{cfg.dag.stencil.steps}"
        + datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    )
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    if not os.path.exists(f"outputs/{run_name}"):
        os.makedirs(f"outputs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    wandb.init(
        project="Stencil Adversarial Data Placement General",
        name=run_name,
        config={
            "env_id": args.env_id,
            "learning_rate": args.learning_rate,
            "num_minibatches": args.num_minibatches,
            "update_epochs": args.update_epochs,
            "clip_coef": args.clip_coef,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "num_iterations": args.num_iterations,
            "graphs_per_update": args.graphs_per_update,
            "reward": args.reward,
            "devices": cfg.system.ngpus,
            "vcus": 1,
            "blocks": cfg.dag.stencil.width,
            "steps": cfg.dag.stencil.steps,
        },
    )

    H, SIM = setup_simulator(
        cfg,
    )
    candidates = SIM.get_mapping_candidates()
    local_graph = SIM.observer.local_graph_features(candidates)
    h = TaskAssignmentNetDeviceOnly(cfg.system.ngpus + 1, args.hidden_dim, local_graph)
    optimizer = torch.optim.Adam(h.parameters(), lr=args.learning_rate)
    rnetmap = RandomNetworkMapper(h)
    H.set_python_mapper(rnetmap)

    h.apply(init_weights)
    # h.load_state_dict(
    #     torch.load(
    #         "/Users/jaeyoung/work/rlgraph_experiments/outputs/ppo_stencil_4x142025-02-12 11:14:03/checkpoint_epoch_2700.pth",
    #         map_location=torch.device("cpu"),
    #         weights_only=True,
    #     )
    # )
    cfg.mapper.type = "block"
    cfg.mapper.python = True
    block_times: List[List[int]] = [] * 9
    for level in range(0, 9):
        for i in range(0, 24):
            cfg.dag.stencil.load_idx = level
            cfg.dag.stencil.permute_idx = i
            h_block, sim_block = setup_simulator(cfg)
            sim_block.run()
            block_time = sim_block.get_current_time()
            print(f"Block: {block_time}")
            block_times[level].append(block_time)
    Hs: List[List[SimulatorHandler]] = [] * 9
    Sims: List[List[Simulator]] = [] * 9
    cfg.env.task_noise = "Lognormal"
    for level in range(0, 9):
        for i in range(0, 24):
            cfg.dag.stencil.load_idx = level
            cfg.dag.stencil.permute_idx = i
            h_block, sim_block = setup_simulator(
                cfg,
                python_mapper=rnetmap,
                randomize_priorities=True,
            )
            Hs[level].append(h_block)
            Sims[level].append(sim_block)

    def collect_batch(episodes, h, global_step=0):
        batch_info = []
        global LEVELS
        LEVELS += 1
        if LEVELS == 9:
            LEVELS = 0
        for e in range(0, episodes):
            sim = Hs[LEVELS][e].copy(Sims[LEVELS][e])
            sim.randomize_priorities()
            done = False
            # Run baseline
            obs, immediate_reward, done, terminated, info = sim.step()
            episode_info = []

            while not done:
                candidates = sim.get_mapping_candidates()
                record = {}
                action_list = RandomNetworkMapper(h).map_tasks(candidates, sim, record)

                obs, immediate_reward, done, terminated, info = sim.step(action_list)
                record["done"] = done
                record["time"] = sim.get_current_time()
                episode_info.append(record)

                if done:
                    baseline = block_times[LEVELS][e]

                    if args.reward == "percent_improvement":
                        record["reward"] = 1 + (baseline - record["time"]) / baseline
                    elif args.reward == "exp":
                        record["reward"] = (
                            math.exp((baseline - record["time"]) / baseline) - 1
                        )
                    elif args.reward == "better":
                        if record["time"] < baseline:
                            record["reward"] = 1
                        else:
                            record["reward"] = 0

                    print(f"{record['reward']}, {record['time']}, Block:{baseline}")
                    break

                else:
                    record["reward"] = 0

            with torch.no_grad():
                for t in range(len(episode_info)):
                    episode_info[t]["returns"] = episode_info[-1]["reward"]
                    episode_info[t]["advantage"] = (
                        episode_info[-1]["reward"] - episode_info[t]["value"]
                    )

            batch_info.extend(episode_info)

        return batch_info

    def batch_update(batch_info, update_epoch, h, optimizer, global_step):
        n_obs = len(batch_info)

        dclipfracs = []

        state = []
        total_rewards = 0
        for i in range(n_obs):
            state.append(batch_info[i]["state"])
            state[i]["dlogprob"] = batch_info[i]["dlogprob"]
            state[i]["value"] = batch_info[i]["value"]
            state[i]["dactions"] = batch_info[i]["dactions"]
            state[i]["advantage"] = batch_info[i]["advantage"]
            state[i]["returns"] = batch_info[i]["returns"]
            total_rewards += batch_info[i]["returns"]

        for k in range(update_epoch):
            loader = torch_geometric.loader.DataLoader(
                state, batch_size=(n_obs // args.num_minibatches), shuffle=True
            )

            for i, batch in enumerate(loader):
                batch: torch_geometric.data.Batch
                out: Tuple[torch.Tensor, torch.Tensor] = h(batch, batch["tasks"].batch)
                d, v = out

                da, dlogprob, dentropy = logits_to_actions(
                    d, batch["dactions"].detach().view(-1)
                )

                dlogratio = dlogprob.view(-1) - batch["dlogprob"].detach().view(-1)
                dratio = dlogratio.exp()

                with torch.no_grad():
                    dold_approx_kl = (-dlogratio).mean()
                    dapprox_kl = ((dratio - 1) - dlogratio).mean()
                    dclipfracs += [
                        ((dratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = batch["advantage"].detach().view(-1)

                dpg_loss1 = mb_advantages * dratio.view(-1)
                dpg_loss2 = mb_advantages * torch.clamp(
                    dratio.view(-1), 1 - args.clip_coef, 1 + args.clip_coef
                )
                dpg_loss = torch.min(dpg_loss1, dpg_loss2).mean()

                newvalue = v.view(-1)
                v_loss_unclipped = (newvalue - batch["returns"].detach().view(-1)) ** 2
                v_clipped = batch["value"].detach().view(-1) + torch.clamp(
                    newvalue - batch["value"].detach().view(-1),
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - batch["returns"].detach().view(-1)) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = dentropy.mean()
                loss = (
                    -1 * (dpg_loss)
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(), args.max_grad_norm)
                optimizer.step()

                wandb.log(
                    {
                        "losses/value_loss": v_loss.item(),
                        "losses/entropy": entropy_loss.item(),
                        "losses/dentropy": dentropy.mean().item(),
                        "losses/dratio": dratio.mean().item(),
                        "losses/dpolicy_loss": dpg_loss.item(),
                        "losses/dold_approx_kl": dold_approx_kl.item(),
                        "losses/dapprox_kl": dapprox_kl.item(),
                        "losses/dclipfrac": np.mean(dclipfracs),
                    },
                    commit=False,
                )
        wandb.log(
            {"episode_reward": total_rewards / n_obs},
        )

    for epoch in range(args.num_iterations):
        print("Epoch: ", epoch)

        batch_info = collect_batch(args.graphs_per_update, h, global_step=epoch)
        batch_update(batch_info, args.update_epochs, h, optimizer, global_step=epoch)

        # --- Gradient Monitoring ---
        total_norm = 0.0
        for name, param in h.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm**2
        total_norm = total_norm**0.5
        # # Save the model
        if (epoch + 1) % 100 == 0:
            torch.save(
                h.state_dict(),
                f"outputs/{run_name}/checkpoint_epoch_{epoch+1}.pth",
            )

    # save pytorch model
    torch.save(h.state_dict(), f"outputs/{run_name}/model.pth")
    wandb.finish()


if __name__ == "__main__":
    my_app()
