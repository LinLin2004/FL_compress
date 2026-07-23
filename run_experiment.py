import argparse
import logging
import pprint
from importlib import import_module
from typing import Any, Dict, List
import os
import datetime

import torch
from torch.utils.data import DataLoader as Dataloader
import yaml
import numpy as np
import random
from multiprocessing import Manager

# Import framework components
from fl_framework.core.coordinator import Coordinator
from fl_framework.core.server import Server
from fl_framework.core.client import HonestClient, ByzantineClient
from fl_framework.data.loader import partition_dataset
from fl_framework.data.samplers import MiniBatchSampler
from fl_framework.utils.functions import seed_all, setup_logging, create_component_from_config


# --- Main Experiment Logic ---

def main():

    parser = argparse.ArgumentParser(description="Byzantine-Robust FL Framework")
    parser.add_argument("--config", "-c", type=str, default="configs_covtype_avg_optimizer_718_tol_1e-3_thr_1e-5_step_10/covtype_param_avg_optimizer_krum_signflipping.yaml", help="Path to the experiment configuration file (YAML)")
    parser.add_argument("--resume", type=str, default=None, help="Path to the checkpoint file", )
    args = parser.parse_args()

    # Load Configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    save_path = config['experiment']['save_path']
    os.makedirs(save_path, exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_path, f"experiment_{now_str}.log")
    setup_logging(config['experiment'].get('log_level', 'INFO'), log_file)

    logging.info("Starting experiment with configuration:")
    logging.info("\n" + pprint.pformat(config))
    # Set Random Seed for Reproducibility
    seed = config['experiment']['seed']
    seed_all(seed)
    
    # Load and Partition Data
    logging.info("Loading and partitioning data...")
    train_dataset = create_component_from_config(config['dataset']['train'])
    test_dataset = create_component_from_config(config['dataset']['test'])

    num_honest = config['honest']['num_clients']
    num_byzantine = config['byzantine']['num_clients']
    total_clients = num_honest + num_byzantine

    client_partitions = partition_dataset(
        dataset=train_dataset,
        num_clients=total_clients,
        **config['dataset']['partition']
    )

    test_dataloader = Dataloader(
        dataset=test_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False
    )

    # Build Core FL Components
    logging.info("Building FL components...")
    
    model = create_component_from_config(config['model'])
    optimizer = create_component_from_config(config['optimizer'])
    train_loss_function = create_component_from_config(config['training']['train_loss_function'])
    test_loss_function = create_component_from_config(config['training']['test_loss_function'])
    metrics = [create_component_from_config(metric_config) for metric_config in config['training']['metrics']]
    aggregator = create_component_from_config(config['aggregator'])

    server = Server(
        model=model,
        aggregator=aggregator,
        optimizer=optimizer,
        train_loss_fn=train_loss_function,
        test_loss_fn=test_loss_function,
        metrics=metrics,
        test_dataloader=test_dataloader,
        devices=config['resources']['gpus']
        # devices = ['cpu']
    )
    
    # Create Clients
    logging.info("Creating clients...")
    clients: List[Any] = []

    # Honest clients
    for id in range(num_honest):
        # print(len(client_partitions[id]))
        sampler = MiniBatchSampler(dataset=client_partitions[id], batch_size=config['dataset']['batch_size'])
        client = HonestClient(client_id=id, data_partition=client_partitions[id], sampler=sampler)
        clients.append(client)

    # Byzantine clients
    for id in range(num_honest, total_clients):
        attack_strategy = create_component_from_config(config['byzantine']['attack'])
        sampler = MiniBatchSampler(dataset=client_partitions[id], batch_size=config['dataset']['batch_size'])
        client = ByzantineClient(
            client_id=id,
            data_partition=client_partitions[id],
            sampler=sampler,
            attack_strategy=attack_strategy
        )
        clients.append(client)

    logging.info(f"Created {len(clients)} clients ({num_honest} Honest, {num_byzantine} Byzantine).")

    # Initialize and Run the Coordinator
    logging.info("Initializing the Coordinator...")
    coordinator = Coordinator(
        server=server,
        clients=clients,
        num_rounds=config['training']['rounds'],
        num_round_steps=config['training']['round_steps'],
        num_byzantine=num_byzantine,
        save_path=save_path
    )

    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        coordinator.resume(args.resume)

    logging.info("Starting the training process...")
    coordinator.run()
    logging.info("Experiment finished successfully.")
if __name__ == "__main__":
    main()

