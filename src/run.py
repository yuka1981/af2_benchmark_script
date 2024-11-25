import argparse
import os
import subprocess
import datetime


def validate_arguments(args):
    if not os.path.isfile(args.input_fasta):
        raise FileNotFoundError(f"Error: Input file '{args.input_fasta}' does not exist.")
    
    if args.preset not in ["monomer", "multimer"]:
        raise ValueError("Error: The 'preset' argument must be 'monomer' or 'multimer'.")
    
    if args.msa not in ["msa", "nomsa"]:
        raise ValueError("Error: The 'msa' argument must be 'msa' or 'nomsa'.")

    if args.platform not in ["amd", "nvidia"]:
        raise ValueError("Error: Supported platforms are 'amd' and 'nvidia'.")

    if args.mode not in ["cpu", "1gpu0", "1gpu1", "2gpus"]:
        raise ValueError("Error: Unsupported mode for the selected platform.")

def setup_environment(args):
    if args.platform == "amd":
        modules = ["hmmer-3.4", "hh-suite-3.3_AVX2", "kalign-3.4.0"]
        for module in modules:
            subprocess.run(["ml", "load", module], check=True)

        if args.mode == "cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
            os.environ["HIP_VISIBLE_DEVICES"] = "-1"
        elif args.mode == "1gpu0":
            os.environ["JAX_PLATFORMS"] = "rocm"
            os.environ["ROCM_PATH"] = "/opt/rocm"
            os.environ["HIP_VISIBLE_DEVICES"] = "0"
        elif args.mode == "1gpu1":
            os.environ["JAX_PLATFORMS"] = "rocm"
            os.environ["ROCM_PATH"] = "/opt/rocm"
            os.environ["HIP_VISIBLE_DEVICES"] = "1"
        elif args.mode == "2gpus":
            os.environ["JAX_PLATFORMS"] = "rocm"
            os.environ["ROCM_PATH"] = "/opt/rocm"
            os.environ["HIP_VISIBLE_DEVICES"] = "0,1"
        else:
            raise ValueError(f"Error: Unsupported mode '{args.mode}' for AMD platform.")

    elif args.platform == "nvidia":
        modules = ["hmmer-3.4-a100", "hh-suite-3.3_AVX2", "kalign-3.4.0-a100"]
        for module in modules:
            subprocess.run(["ml", "load", module], check=True)

        if args.mode == "cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif args.mode == "1gpu0":
            os.environ["JAX_PLATFORMS"] = "cuda"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        elif args.mode == "1gpu1":
            os.environ["JAX_PLATFORMS"] = "cuda"
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        elif args.mode == "2gpus":
            os.environ["JAX_PLATFORMS"] = "cuda"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        else:
            raise ValueError(f"Error: Unsupported mode '{args.mode}' for NVIDIA platform.")

    else:
        raise ValueError("Error: Unknown platform.")

    if "CONDA_PREFIX" not in os.environ or "CONDA_DEFAULT_ENV" not in os.environ:
        raise EnvironmentError("Error: Environment variables CONDA_PREFIX or CONDA_DEFAULT_ENV are missing.")
    os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"


def run_alphafold(args):
    base_name = os.path.splitext(os.path.basename(args.input_fasta))[0]
    hostname = os.uname().nodename
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
    output_dir = f"{os.getcwd()}/output/{hostname}-{os.environ['CONDA_DEFAULT_ENV']}-{timestamp}/{base_name}"
    os.makedirs(output_dir, exist_ok=True)

    msa_setting = args.msa == "msa"
    preset_args = []
    if args.preset == "monomer":
        preset_args = [
            "--model_preset=monomer",
            "--db_preset=full_dbs",
            "--uniref90_database_path=/data/uniref90.fasta",
        ]
    elif args.preset == "multimer":
        preset_args = [
            "--model_preset=multimer",
            "--db_preset=full_dbs",
            "--uniprot_database_path=/data/uniprot.fasta",
        ]

    common_args = [
        f"--fasta_paths={args.input_fasta}",
        f"--output_dir={output_dir}",
        "--max_template_date=2022-01-01",
        f"--use_precomputed_msas={str(msa_setting).lower()}",
    ]

    command = ["python3", "run_alphafold.py"] + common_args + preset_args
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run AlphaFold2 with specified settings.")
    parser.add_argument("input_fasta", help="Input FASTA file path.")
    parser.add_argument("description", help="Job description.")
    parser.add_argument("mode", help="Execution mode: cpu, 1gpu0, 1gpu1, 2gpus.")
    parser.add_argument("preset", help="Model preset: monomer or multimer.")
    parser.add_argument("platform", help="Platform: amd or nvidia.")
    parser.add_argument("msa", help="MSA setting: msa or nomsa.")
    args = parser.parse_args()

    try:
        validate_arguments(args)
        setup_environment(args)
        run_alphafold(args)
    except Exception as e:
        print(e)
        exit(1)


if __name__ == "__main__":
    main()

