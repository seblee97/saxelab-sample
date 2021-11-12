import run
import submitit

executor = submitit.AutoExecutor(folder="log_exp")

executor.update_parameters(
    timeout_min=20,
    mem_gb=30,
    gpus_per_node=1,
    cpus_per_task=8,
    slurm_array_parallelism=256,
    slurm_partition="gpu",
)

jobs = []
with executor.batch():
    for seed in range(1):
        job = executor.submit(run.run, num_epochs=5, batch_size=32, lr=0.001)
