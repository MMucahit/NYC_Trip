from prefect.filesystems import LocalFileSystem
from prefect.infrastructure import Process

my_storage_block = LocalFileSystem(
    basepath="./"
)

my_storage_block.save(
    name="nyctaxi-storage-block"
)

my_process_infra = Process(working_dir="./")

my_process_infra.save(
    name="nyctaxi-process-infra",
    overwrite=True    
)
