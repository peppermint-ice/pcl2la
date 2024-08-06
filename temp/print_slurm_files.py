for model in [
    "elastic_net_kf2",
    "elastic_net_kf2_byyear",
    "lasso_kf2",
    "lasso_kf2_byyear",
    "linear_kf2",
    "linear_kf2_byyear",
    "mlp_kf2",
    "mlp_kf2_byyear",
    "ridge_kf2",
    "ridge_kf2_byyear",
    "xgb_kf2",
    "xgb_kf2_byyear",
    "rf_kf2",
    "rf_kf2_byyear",
]:
  print(f'#!/bin/bash\n#SBATCH --job-name={model}\n#SBATCH --output=/sci/labs/davidhelman/d_usenko/slurm_jobs/stats/{model}_%A_%a.out\n#SBATCH --error=/sci/labs/davidhelman/d_usenko/slurm_jobs/stats/{model}_%A_%a.err\n#SBATCH --array=1-442\n#SBATCH --time=24:00:00\n#SBATCH --mem=4G\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=1\n\n# Load env\nsource /sci/labs/davidhelman/d_usenko/venvs/venv_dbscanning_310/bin/activate.csh\n# Set env path\nsetenv PYTHONPATH /sci/labs/davidhelman/d_usenko/pcl2la\n\n# Load necessary modules\nmodule purge\nmodule load python/3.8.6\n\n# Define the directory containing the CSV files and the Python script\nDATA_DIR="/sci/archive/davidhelman/d_usenko/plys/data/ready_for_training"\nSCRIPT_DIR="/sci/labs/davidhelman/d_usenko/pcl2la/04_ml"\n\n# Navigate to the script directory\ncd $SCRIPT_DIR || exit\n\n# Get the CSV file corresponding to the array task ID\nCSV_FILE=$(ls $DATA_DIR | sed -n "${{SLURM_ARRAY_TASK_ID}}p")\n\n# Run the Python script with the CSV file\npython slurm_{model}.py "$DATA_DIR/$CSV_FILE"')
  print(f'\n\n\n\n\n')

  for model in [
      "elastic_net_kf_noyear",
      "elastic_net_kf_byyear",
      "lasso_kf_noyear",
      "lasso_kf_byyear",
      "linear_kf_noyear",
      "linear_kf_byyear",
      "mlp_kf_noyear",
      "mlp_kf_byyear",
      "ridge_kf_noyear",
      "ridge_kf_byyear",
      "xgb_kf_noyear",
      "xgb_kf_byyear",
      "rf_kf_noyear",
      "rf_kf_byyear",
  ]:
      print(f'sbatch /sci/labs/davidhelman/d_usenko/slurm_jobs/job_{model}.sh')


