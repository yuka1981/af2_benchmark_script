#!/bin/bash

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 <input_fasta> <description> <cpu/gpu> <monomer/multimer> <nvidia/amd> <nomsa/msa>"
    exit 1
fi

input=$1
description=$2
mode=$3
preset=$4
platform=$5
msa=$6

# Validate input arguments and file existence
if [[ ! -f "$input" ]]; then
    echo "Error: Input file '$input' does not exist."
    exit 1
fi

if [[ "$preset" != "monomer" && "$preset" != "multimer" ]]; then
    echo "Error: The fourth argument must be 'monomer' or 'multimer'."
    exit 1
fi

# MSA setting
if [[ "$msa" == "msa" ]]; then
    msa_setting=true
elif [[ "$msa" == "nomsa" ]]; then
    msa_setting=false
else 
    echo "Error: No MSA setting. Existing script."
    exit 1
fi

# Purge and load modules
ml purge

case $host_name in
    gn502 | gn503)
        # Nvidia Grace cpu
        ml load hmmer-3.4-arm 
        ml load hh-suite-3.3_aarch64_native
        ml load kalign-3.4.0_aarch64_native
        ml load nvhpc/24.5
        ;;
    MI210 | gn11)
        # AMD cpu
        ml load hmmer-3.4-a100 
        ml load hh-suite-3.3_AVX2
        ml load kalign-3.4.0-a100
        ;;
    *)
        # Intel cpu w/ gcc
        ml load hmmer-3.4 suite-3.3_AVX2 kalign-3.4.0
        ;;
esac


# Combined AMD and CUDA platform setup
if [[ "$platform" == "amd" ]]; then
    case $mode in
        cpu)
            export JAX_PLATFORMS=cpu
            export HIP_VISIBLE_DEVICES=-1
            ;;
        1gpu0)
            export JAX_PLATFORMS=rocm
            export ROCM_PATH=/opt/rocm
            export HIP_VISIBLE_DEVICES=0
            ;;
        1gpu1)
            export JAX_PLATFORMS=rocm
            export ROCM_PATH=/opt/rocm
            export HIP_VISIBLE_DEVICES=1
            ;;
        2gpus)
            export JAX_PLATFORMS=rocm
            export ROCM_PATH=/opt/rocm
            export HIP_VISIBLE_DEVICES=0,1
            ;;
        *)
            echo "Error: Unsupported mode for AMD platform: $mode"
            exit 1
            ;;
    esac
elif [[ "$platform" == "nvidia" ]]; then
    case $mode in
        cpu)
            export JAX_PLATFORMS=cpu
            export CUDA_VISIBLE_DEVICES=-1
            ;;
        1gpu0)
            export JAX_PLATFORMS=cuda
            export CUDA_VISIBLE_DEVICES=0
            ;;
        1gpu1)
            export JAX_PLATFORMS=cuda
            export CUDA_VISIBLE_DEVICES=1
            ;;
        *)
            echo "Error: Unsupported mode for NVIDIA platform: $mode"
            exit 1
            ;;
    esac
else
    echo "Error: Unknown platform. Supported platforms are 'amd' and 'nvidia'."
    exit 1
fi

# Check conda environment variables
if [[ -z "$CONDA_PREFIX" || -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "Error: Environment variables CONDA_PREFIX or CONDA_DEFAULT_ENV are empty. Exiting script."
    exit 1
else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

    base_name=$(echo $input | sed 's/.fasta//g')
    host_name=$(hostname | sed 's/.openlab//g')
    date_time=$(date +%m%d-%H%M%S)
    work_dir=$(echo $PWD)
    output_dir="$work_dir/output/$host_name-$CONDA_DEFAULT_ENV-$date_time"

    mkdir -p "$output_dir/$base_name"

    if [[ "$msa" == "msa" ]]; then
        if [[ -d "$work_dir/output/${base_name}_msa"  ]];then
            cp -r $work_dir/output/${base_name}_msa/* $output_dir/$base_name
        else
            echo "Error: No MSA folder existed. Exiting script."
            exit 1
        fi
    fi

    numactl_exec="numactl --cpunodebind=0 --membind=0 python3"
fi

# Set up paths and variables
case $host_name in
    gn503)
	data_dir=/mlperf41/alphafold_data
        ;;
    MI210)
	data_dir=/data/alphafold_data
        ;;
    *)
	data_dir=/lustre2/reidlin/alphafold_data_hot
        ;;
esac

declare -A db_paths=(
    [bfd]="$data_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
    [uniref90]="$data_dir/uniref90/uniref90.fasta"
    [mgnify]="$data_dir/mgnify/mgy_clusters_2022_05.fa"
    [pdb70]="$data_dir/pdb70/pdb70"
    [uniprot]="$data_dir/uniprot/uniprot.fasta"
    [pdb_seqres]="$data_dir/pdb_seqres/pdb_seqres.txt"
    [template_mmcif]="$data_dir/pdb_mmcif/mmcif_files"
    [obsolete_pdbs]="$data_dir/pdb_mmcif/obsolete.dat"
    [uniref30]="$data_dir/uniref30/UniRef30_2021_03"
    [small_bfd]="$data_dir/small_bfd/bfd-first_non_consensus_sequences.fasta"
)

max_template_date='2022-01-01'
log="$output_dir/$base_name-$date_time.log"

# Log start
echo "$base_name AlphaFold2 start at $(date +%m%d-%H:%M:%S) time_stamp: $(date +%s)" > "$log"

# Common arguments for running AlphaFold
common_args=(
    --fasta_paths="$input"
    --data_dir="$data_dir"
    --output_dir="$output_dir"
    --max_template_date="$max_template_date"
    --obsolete_pdbs_path="${db_paths[obsolete_pdbs]}"
    --uniref30_database_path="${db_paths[uniref30]}"
    --bfd_database_path="${db_paths[bfd]}"
    --use_precomputed_msas=$msa_setting
    --use_gpu_relax=false
    --models_to_relax=none
)

# Set preset-specific arguments
if [[ "$preset" == "monomer" ]]; then
    preset_args=(
        --model_preset=monomer
        --db_preset=full_dbs
        --uniref90_database_path="${db_paths[uniref90]}"
        --mgnify_database_path="${db_paths[mgnify]}"
        --template_mmcif_dir="${db_paths[template_mmcif]}"
        --pdb70_database_path="${db_paths[pdb70]}"
    )
elif [[ "$preset" == "multimer" ]]; then
    preset_args=(
        --model_preset=multimer
        --db_preset=full_dbs
        --uniprot_database_path="${db_paths[uniprot]}"
        --pdb_seqres_database_path="${db_paths[pdb_seqres]}"
        --uniref90_database_path="${db_paths[uniref90]}"
        --mgnify_database_path="${db_paths[mgnify]}"
        --template_mmcif_dir="${db_paths[template_mmcif]}"
    )
else
    echo "Error: Invalid preset '$preset'."
    exit 1
fi

# Run AlphaFold
$numactl_exec "$work_dir/run_alphafold.py" "${common_args[@]}" "${preset_args[@]}" 2>&1 | tee -a "$log"

# Log finish
echo "$base_name AlphaFold2 finish at $(date +%m%d-%H:%M:%S) time_stamp: $(date +%s)" >> "$log"

