#! /bin/bash

nV=30
nVtilde=0
V_bar=1.0
Vtilde_bar=1.0
sigma_V_norm=0.132
sigma_Vtilde_norm=0.0

if (( $(echo "$sigma_Vtilde_norm == 0.0" |bc -l) )); then
    domain_folder='WZV'
    mkdir -p ./job-outs/$domain_folder
    mkdir -p ./bash/$domain_folder
elif (( $(echo "$sigma_V_norm == 0.0" |bc -l) )); then
    domain_folder='WZVtilde'
    mkdir -p ./job-outs/$domain_folder
    mkdir -p ./bash/$domain_folder
fi

for chiUnderline in 0.5
do
    for a_e in 0.14
    do
        for a_h in -1
        do
            for gamma_e in 4.0
            do
                for gamma_h in 6.0 8.0
                do
                    for psi_e in 1.0
                    do
                        for psi_h in 1.0
                        do


# for chiUnderline in 0.5
# do
#     for a_e in 0.15
#     do
#         for a_h in 0.13 0.10 0.05
#         do
#             for gamma_e in 2.0 3.0 4.0
#             do
#                 for gamma_h in 6.0 8.0
#                 do
#                     for psi_e in 1.0
#                     do
#                         for psi_h in 1.0
#                         do
                            model_folder=chiUnderline_${chiUnderline}_a_e_${a_e}_a_h_${a_h}_gamma_e_${gamma_e}_gamma_h_${gamma_h}_psi_e_${psi_e}_psi_h_${psi_h}
                            mkdir -p ./job-outs/$domain_folder/$model_folder
                            mkdir -p ./bash/$domain_folder/$model_folder

                            touch ./bash/$domain_folder/$model_folder/plot.sh
                            tee ./bash/$domain_folder/$model_folder/plot.sh << EOF
#! /bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=plot
#SBATCH --output=./job-outs/$domain_folder/$model_folder/plot.out
#SBATCH --error=./job-outs/$domain_folder/$model_folder/plot.err
#SBATCH --time=0-10:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=2000

module load python/anaconda-2021.05

python3 /project/lhansen/mfrSuite_Ela_midway3/SolvedModels/plot.py --chiUnderline ${chiUnderline} --a_e ${a_e} --a_h ${a_h} --gamma_e ${gamma_e} --gamma_h ${gamma_h} --psi_e ${psi_e} --psi_h ${psi_h} \
                                                    --nV ${nV} --nVtilde ${nVtilde} --V_bar ${V_bar} --Vtilde_bar ${Vtilde_bar} --sigma_V_norm ${sigma_V_norm} --sigma_Vtilde_norm ${sigma_Vtilde_norm} \

EOF
                            sbatch ./bash/$domain_folder/$model_folder/plot.sh
                        done
                    done
                done
            done
        done
    done
done
