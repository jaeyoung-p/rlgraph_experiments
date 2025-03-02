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
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator):
        data = simulator.observer.local_graph_features(candidates, k_hop=1)
        data = data.to(self.device)
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

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator, output=None):
        data = simulator.observer.local_graph_features(candidates, k_hop=1)
        data = data.to(self.device)
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
        obs = obs.to(self.device)
        p, d, v = self.model.forward(obs)
        _, plogprob, pentropy = logits_to_actions(p, paction)
        _, dlogprob, dentropy = logits_to_actions(d, daction)
        return (p, plogprob, pentropy), (d, dlogprob, dentropy), v


def get_random_mapper(
    cfg, args, device, trained_model: Optional[str] = None
) -> tuple[TaskAssignmentNetDeviceOnly, RandomNetworkMapper]:
    _, preSim = setup_simulator(
        cfg,
    )
    candidates = preSim.get_mapping_candidates()
    local_graph = preSim.observer.local_graph_features(candidates)
    h = TaskAssignmentNetDeviceOnly(
        cfg.system.ngpus + 1, args.hidden_dim, local_graph
    ).to(device)
    optimizer = torch.optim.Adam(h.parameters(), lr=args.learning_rate)
    rnetmap = RandomNetworkMapper(h, device)
    if trained_model is not None:
        h.load_state_dict(
            torch.load(
                trained_model,
                map_location=device,
                weights_only=True,
            )
        )
    else:
        h.apply(init_weights)
    return h, rnetmap, optimizer


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def my_app(cfg: DictConfig) -> None:
    args = Args()
    run_name = (
        f"ppo_{args.env_id}_{cfg.dag.stencil.width}x{cfg.dag.stencil.steps}_"
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
        project="Stencil Adversarial Data Placement Single",
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
            "devices": cfg.system.ngpus,
            "vcus": 1,
            "blocks": cfg.dag.stencil.width,
            "steps": cfg.dag.stencil.steps,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h, rnetmap, optimizer = get_random_mapper(cfg, args, device)

    baseH, baseSim = setup_simulator(
        cfg,
        python_mapper=rnetmap,
        randomize_durations=False,
        randomize_priorities=False,
        log=False,
    )

    eftH, eftSim = setup_simulator(
        cfg,
        force_eft_dequeue=True,
    )
    eftSim.run()
    eftTime = eftSim.get_current_time()

    blockH, blockSim = setup_simulator(
        cfg,
        python_mapper="block",
    )
    blockSim.run()
    blockTime = blockSim.get_current_time()
    print(f"EFT: {eftTime}, Block: {blockTime}")

    def collect_batch(episodes):
        batch_info = []
        compareMetrics = {
            "vsEFT": 0,
            "vsBlock": 0,
            "vsOptimal": 0,
        }
        for e in range(0, episodes):
            sim = baseH.copy(baseSim)
            done = False
            obs, immediate_reward, done, terminated, info = sim.step()
            episode_info = []
            if not isinstance(sim.pymapper, RandomNetworkMapper):
                raise ValueError(f"Training with {type(sim.pymapper)} mapper")
            while not done:
                candidates = sim.get_mapping_candidates()
                record = {}
                action_list = sim.pymapper.map_tasks(candidates, sim, record)

                obs, immediate_reward, done, terminated, info = sim.step(action_list)
                record["done"] = done
                if done:
                    current_time = sim.get_current_time()
                    optimal = (
                        (cfg.dag.stencil.width**2) * cfg.dag.stencil.steps * 1000 / 4
                    )
                    record["reward"] = 1 + (optimal - current_time) / optimal
                    print(
                        f"{record['reward']}, {current_time}, optimal:{optimal}, Block:{blockTime}, EFT:{eftTime}"
                    )
                    compareMetrics["vsEFT"] += eftTime / current_time
                    compareMetrics["vsBlock"] += blockTime / current_time
                    compareMetrics["vsOptimal"] += optimal / current_time
                else:
                    record["reward"] = 0
                episode_info.append(record)

            with torch.no_grad():
                for t in range(len(episode_info)):
                    episode_info[t]["reward"] = episode_info[-1]["reward"]
                    episode_info[t]["advantage"] = (
                        episode_info[-1]["reward"] - episode_info[t]["value"]
                    )

            batch_info.extend(episode_info)
        averageMetrics = {k: v / episodes for k, v in compareMetrics.items()}
        wandb.log(averageMetrics, commit=False)
        return batch_info

    def batch_update(batch_info, update_epoch, h, optimizer):
        n_obs = len(batch_info)
        state = []
        total_rewards = 0
        for i in range(n_obs):
            state.append(batch_info[i]["state"])
            state[i]["dlogprob"] = batch_info[i]["dlogprob"]
            state[i]["value"] = batch_info[i]["value"]
            state[i]["dactions"] = batch_info[i]["dactions"]
            state[i]["advantage"] = batch_info[i]["advantage"]
            state[i]["reward"] = batch_info[i]["reward"]
            total_rewards += batch_info[i]["reward"]
        accum_metrics = {
            "losses/value_loss": 0.0,
            "losses/entropy": 0.0,
            "losses/dentropy": 0.0,
            "losses/dratio": 0.0,
            "losses/dpolicy_loss": 0.0,
            "losses/dold_approx_kl": 0.0,
            "losses/dapprox_kl": 0.0,
            "losses/dclipfrac": 0.0,
        }
        num_batches = 0
        for k in range(update_epoch):
            loader = torch_geometric.loader.DataLoader(
                state, batch_size=(n_obs // args.num_minibatches), shuffle=True
            )

            for i, batch in enumerate(loader):
                batch = batch.to(device)
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
                    dclipfrac = (
                        ((dratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = batch["advantage"].detach().view(-1)

                dpg_loss1 = mb_advantages * dratio.view(-1)
                dpg_loss2 = mb_advantages * torch.clamp(
                    dratio.view(-1), 1 - args.clip_coef, 1 + args.clip_coef
                )
                dpg_loss = torch.min(dpg_loss1, dpg_loss2).mean()

                newvalue = v.view(-1)
                v_loss_unclipped = (newvalue - batch["reward"].detach().view(-1)) ** 2
                v_clipped = batch["value"].detach().view(-1) + torch.clamp(
                    newvalue - batch["value"].detach().view(-1),
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - batch["reward"].detach().view(-1)) ** 2
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

                accum_metrics["losses/value_loss"] += v_loss.item()
                accum_metrics["losses/entropy"] += entropy_loss.item()
                accum_metrics["losses/dentropy"] += dentropy.mean().item()
                accum_metrics["losses/dratio"] += dratio.mean().item()
                accum_metrics["losses/dpolicy_loss"] += dpg_loss.item()
                accum_metrics["losses/dold_approx_kl"] += dold_approx_kl.item()
                accum_metrics["losses/dapprox_kl"] += dapprox_kl.item()
                accum_metrics["losses/dclipfrac"] += dclipfrac
                num_batches += 1

        avg_metrics = {k: v / num_batches for k, v in accum_metrics.items()}
        wandb.log(avg_metrics, commit=False)
        wandb.log({"total_rewards": total_rewards / n_obs})

    for epoch in range(args.num_iterations):
        print("Epoch: ", epoch)

        batch_info = collect_batch(args.graphs_per_update)
        batch_update(batch_info, args.update_epochs, h, optimizer)

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
