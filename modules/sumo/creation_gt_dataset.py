from .creation_agg_dataset import create_agg_dataset

def create_gt_dataset(
        output_dir,
        epoch_index,
        sumo_simulator,
        num_replications
):
    for rep in range(num_replications):
        sumo_simulator.simulation(epoch_index=epoch_index,rep=rep)
    create_agg_dataset(output_dir=output_dir,epoch_index=epoch_index,num_replications=num_replications)