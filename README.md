# Dataset Filtering and Training Dataset Generation

This script generates a training dataset for a machine learning model based on the cooking metrics, faulty intervals,
and batch registry of an arepa production line. It filters the cooking metrics based on the specified machine,
arepa type, and time range, removes the faulty intervals, and calculates the hourly average metrics for the specified
conditions. The final dataset is saved to `output_dataset/training_dataset.csv`.

# Requirements

- Python 3.6+
- Pandas library

## Usage

To use the script, follow the steps below:

1. Clone the repository:

```bash
git clone
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python arepas.py --cooking_metrics input_datasets/cooking_metrics.csv --faulty_intervals input_datasets/faulty_intervals.csv --batch_registry input_datasets/batch_registry.csv --machine m1 --arepa_type a1 --start_time 2020-11-01T00:00:00 --end_time 2020-11-02T23:59:59 --output output_dataset/training_dataset.csv
```

Replace
the `--cooking_metrics`, `--faulty_intervals`, `--batch_registry`, `--machine`, `--arepa_type`, `--start_time`, `--end_time`,
and `--output` arguments with the desired values.

Or you can use `-cm`, `-fi`, `-br`, `-m`, `-at`, -`st`, `-et`, and `-o` as the short version of the arguments.

```bash
python arepas.py -cm input_datasets/cooking_metrics.csv -fi input_datasets/faulty_intervals.csv -br input_datasets/batch_registry.csv -m m1 -at a1 -st 2020-11-01T00:00:00 -et 2020-11-02T23:59:59 -o output_dataset/training_dataset.csv
```

# Arguments

The following arguments are available:

- `--cooking_metrics`, `-cm`: Path to the cooking metrics file (default: `input_dataset/cooking_metrics.csv`).
- `--faulty_intervals`, `-fi`: Path to the faulty intervals file (default: `input_dataset/faulty_intervals.csv`).
- `--batch_registry`, `-br`: Path to the batch registry file (default: `input_dataset/batch_registry.csv`).
- `--machine`, `-m`: Machine ID.
- `--arepa_type`, `-at`: Arepa type.
- `--start_time`, `-st`: Start time.
- `--end_time`, `-et`: End time.
- `--output`, `-o`: Output file path.

# Output

The script saves the final dataset to the specified output file path.
