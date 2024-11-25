#!/bin/bash

# Input file
input_file=$1

# Initialize variables
start_time=""
end_time=""
msa_start_time=""
inference_end_time=""
jackhmmer_uniref90="N/A"
jackhmmer_mgy_cluster="N/A"
hhsearch="N/A"
hhblits="N/A"
model_times=()

# Extract and calculate times
start_time=$(grep 'AlphaFold2 start at' "$input_file" | cut -d ' ' -f7)
end_time=$(grep 'AlphaFold2 finish at' "$input_file" | cut -d ' ' -f7)


if [[ -n $start_time && -n $end_time ]]; then
    #start_seconds=$(date -d "1970-01-01 $start_time" +%s)
    #end_seconds=$(date -d "1970-01-01 $end_time" +%s)
    diff_seconds=$((end_time - start_time))

    formatted_time=$(awk -v seconds="$diff_seconds" '
    BEGIN {
        hh = int(seconds / 3600);
        mm = int((seconds % 3600) / 60);
        ss = seconds % 60;
        printf "%02d:%02d:%02d", hh, mm, ss;
    }
    ')
#    hh=$(printf "%02d" $((diff_seconds / 3600)))
#    mm=$(printf "%02d" $(( (diff_seconds % 3600) / 60 )))
#    ss=$(printf "%02d" $((diff_seconds % 60)))
#    formatted_time="$hh:$mm:$ss"
else
    diff_seconds="N/A"
    formatted_time="N/A"
fi

# Parse input file line by line
while IFS= read -r line; do
    case "$line" in
        *"Finished Jackhmmer (uniref90.fasta)"*)
            jackhmmer_uniref90=$(echo "$line" | grep -oP '\d+\.\d+(?= seconds)')
            ;;
        *"Finished Jackhmmer (mgy_clusters_2022_05.fa)"*)
            jackhmmer_mgy_cluster=$(echo "$line" | grep -oP '\d+\.\d+(?= seconds)')
            ;;
        *"Finished HHsearch query"*)
            hhsearch=$(echo "$line" | grep -oP '\d+\.\d+(?= seconds)')
            ;;
        *"Finished HHblits query"*)
            hhblits=$(echo "$line" | grep -oP '\d+\.\d+(?= seconds)')
            ;;
        *"Searching for template for"*)
            msa_start_time=$(echo "$line" | grep -oP '\d{2}:\d{2}:\d{2}')
            ;;
        *"Total JAX model model"*)
            model_times+=("$(echo "$line" | grep -oP '\d+\.\d+(?=s)')")
            ;;
        *"AlphaFold2 finish"*)
            inference_end_time=$(echo "$line" | grep -oP '\d{2}:\d{2}:\d{2}')
            ;;
    esac
done < "$input_file"

# Initialize calculation results
msa_total_sec="N/A"
inference_total_sec="N/A"
io_others="N/A"

if [[ -n $inference_end_time ]]; then
    inference_total_sec=$(echo "${model_times[@]}" | awk '{for(i=1;i<=NF;i++) sum+=$i} END{print sum}')
else
    echo "One or more required variables are empty. Skipping inference_total_sec calculation."
fi

# Only calculate inference_total_sec if all required variables are non-empty
if [[ -n $msa_start_time && -n $inference_total_sec && "$hhblits" != "N/A" && "$jackhmmer_uniref90" != "N/A" && "$jackhmmer_mgy_cluster" != "N/A" ]]; then
    msa_total_sec=$(echo "$hhblits + $jackhmmer_uniref90 + $jackhmmer_mgy_cluster" | bc)
    inference_total_sec=$(echo "${model_times[@]}" | awk '{for(i=1;i<=NF;i++) sum+=$i} END{print sum}')
    if [[ "$msa_total_sec" != "N/A" && "$inference_total_sec" != "N/A" ]]; then
        io_others=$(echo "$diff_seconds - $msa_total_sec - $inference_total_sec" | bc)
    fi
fi

# Output results
echo -e "wall(hh:mm::ss) wall(secs) msa inference io_others Jackhmmer(uniref90) Jackhmmer(mgy_cluster) HHsearch HHblits model_times"
echo -e "$formatted_time $diff_seconds $msa_total_sec $inference_total_sec $io_others $jackhmmer_uniref90 $jackhmmer_mgy_cluster $hhsearch $hhblits ${model_times[@]}"

