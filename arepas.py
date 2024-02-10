import logging
from argparse import ArgumentParser as ArgumentParser
from datetime import datetime
from pandas import (read_csv as pd_import_csv, merge as pd_merge, Grouper as pd_Grouper, to_datetime as pd_to_datetime,
                    DataFrame as pandas_DataFrame)
from logging import getLogger

logger = getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def load_dataset(file_path: str, date_columns: [str], delimiter: str = ';',
                 decimal_separator: str = ',') -> pandas_DataFrame:
    """
    Load the dataset from the specified file path
    :param file_path: Path to the file containing the dataset
    :type file_path: str
    :param date_columns: List of columns containing date values
    :type date_columns: [str]
    :param delimiter: Delimiter used in the file
    :type delimiter: str
    :param decimal_separator: Decimal separator used in the file
    :type decimal_separator: str
    :return: DataFrame containing the loaded dataset
    :rtype: pandas_DataFrame
    """

    try:
        file_dataset = pd_import_csv(file_path, delimiter=delimiter, decimal=decimal_separator)
    except FileNotFoundError as file_not_found:
        logger.error(f"Error loading the file: {file_not_found}")
        raise file_not_found

    for col in date_columns:
        if col in file_dataset.columns:
            file_dataset[col] = pd_to_datetime(file_dataset[col], errors='coerce')

    return file_dataset


def filter_cooking_data(cooking_metrics: pandas_DataFrame, machine: str, start_time: datetime,
                        end_time: datetime) -> pandas_DataFrame:
    """
    Filter the cooking metrics based on the specified conditions
    :param cooking_metrics: DataFrame containing the cooking metrics
    :type cooking_metrics: pandas_DataFrame
    :param machine: Machine ID
    :type machine: str
    :param start_time: Start time
    :type start_time: datetime
    :param end_time: End time
    :type end_time: datetime
    :return: DataFrame containing the filtered cooking metrics
    :rtype: pandas_DataFrame
    """

    return cooking_metrics[(cooking_metrics['machine_id'] == machine) &
                           (cooking_metrics['timestamp'] >= start_time) &
                           (cooking_metrics['timestamp'] <= end_time)]


def filter_faulty_intervals(faulty_intervals: pandas_DataFrame, filtered_metrics: pandas_DataFrame,
                            machine_id: str) -> pandas_DataFrame:
    """
    Filter the faulty intervals from the cooking metrics
    :param faulty_intervals: DataFrame containing the faulty intervals
    :type faulty_intervals: pandas_DataFrame
    :param filtered_metrics: DataFrame containing the filtered cooking metrics
    :type filtered_metrics: pandas_DataFrame
    :param machine_id: Machine ID
    :type machine_id: str
    :return: DataFrame containing the filtered cooking metrics
    :rtype: pandas_DataFrame
    """

    for _, row in faulty_intervals.iterrows():
        if row['machine_id'] == machine_id:
            faulty_mask = ((filtered_metrics['timestamp'] >= row['start_time']) &
                           (filtered_metrics['timestamp'] <= row['end_time']))
            filtered_metrics = filtered_metrics[~faulty_mask]

    return filtered_metrics


def group_by_hourly_average_cooking_metrics(merged_data: pandas_DataFrame, time_column: str = 'timestamp',
                                            frequency: str = 'h') -> pandas_DataFrame:
    """
    Group the merged data by hourly average metrics
    :param merged_data: DataFrame containing the merged data
    :type merged_data: pandas_DataFrame
    :param time_column: Name of the column containing the timestamp
    :type time_column: str
    :param frequency: Frequency for the grouping
    :type frequency: str
    :return: DataFrame containing the grouped data
    :rtype: pandas_DataFrame
    """

    hourly_avg_metrics = merged_data.groupby([pd_Grouper(key=time_column, freq=frequency), 'arepa_type']).agg(
        {'metric_1': 'mean',
         'metric_2': 'mean'}).reset_index()
    return hourly_avg_metrics


def filter_by_arepa_type(merged_data: pandas_DataFrame, arepa: str) -> pandas_DataFrame:
    """
    Filter the merged data for the specific arepa type
    :param merged_data: DataFrame containing the merged data
    :type merged_data: pandas_DataFrame
    :param arepa: Arepa type
    :type arepa: str
    :return: DataFrame containing the filtered data
    :rtype: pandas_DataFrame
    """

    return merged_data[(merged_data['arepa_type'] == arepa)]


def generate_training_dataset(cooking_path: str, faulty_path: str, batch_path: str, machine_id: str,
                              arepa_type_name: str, start_time: str, end_time: str) -> pandas_DataFrame:
    """
    Generate the training dataset
    :param cooking_path: Path to the cooking metrics file
    :type cooking_path: str
    :param faulty_path: Path to the faulty intervals file
    :type faulty_path: str
    :param batch_path: Path to the batch registry file
    :type batch_path: str
    :param machine_id: Machine ID
    :type machine_id: str
    :param arepa_type_name: Arepa type
    :type arepa_type_name: str
    :param start_time: Start time
    :type start_time: str
    :param end_time: End time
    :type end_time: str
    :return: DataFrame containing the training dataset
    :rtype: pandas_DataFrame
    """

    # Load the datasets
    cooking_metrics = load_dataset(file_path=cooking_path, date_columns=['timestamp', 'start_time', 'end_time'])
    logger.info(f"Cooking metrics: \n{cooking_metrics}\n")
    faulty_intervals = load_dataset(file_path=faulty_path, date_columns=['timestamp', 'start_time', 'end_time'])
    logger.info(f"Faulty intervals: \n{faulty_intervals}\n")
    batch_registry = load_dataset(file_path=batch_path, date_columns=['timestamp', 'start_time', 'end_time'])
    logger.info(f"Batch registry: \n{batch_registry}\n")

    # Convert start and end time to datetime
    start_datetime = datetime.fromisoformat(start_time)
    end_datetime = datetime.fromisoformat(end_time)

    # Filter out the cooking metrics based on the specified conditions
    filtered_metrics = filter_cooking_data(cooking_metrics=cooking_metrics, machine=machine_id,
                                           start_time=start_datetime,
                                           end_time=end_datetime)
    logger.info(
        f"Cooking metrics filtered by timestamp between {start_datetime} and {end_datetime}: \n{filtered_metrics}\n")

    # Filter out the faulty intervals from the cooking metrics
    filtered_metrics = filter_faulty_intervals(faulty_intervals=faulty_intervals, filtered_metrics=filtered_metrics,
                                               machine_id=machine_id)
    logger.info(f"Cooking metrics filtered by timestamp and faulty intervals: \n{filtered_metrics}\n")

    # Merge the filtered metrics with the batch registry
    merged_data = pd_merge(filtered_metrics, batch_registry, on='batch_id')
    logger.info(
        f"Cooking metrics filtered by timestamp and faulty intervals merged with batch registry: \n{merged_data}\n")

    # Group the merged data by hourly average metrics
    hourly_avg_metrics = group_by_hourly_average_cooking_metrics(merged_data=merged_data)
    logger.info(f"Hourly average metrics: \n{hourly_avg_metrics}\n")

    # Filter the merged data for the specific arepa type
    result_dataset = filter_by_arepa_type(merged_data=hourly_avg_metrics, arepa=arepa_type_name)
    logger.info(f"Final dataset filtered by arepa_type: \n{result_dataset}\n")

    return result_dataset


if __name__ == '__main__':
    logger.info("Starting training dataset generation")

    parser = ArgumentParser()
    parser.add_argument('--cooking_metrics', '-cm', help="cooking metrics file path", type=str,
                        default='input_dataset/cooking_metrics.csv')
    parser.add_argument('--faulty_intervals', '-fi', help="faulty intervals file path", type=str,
                        default='input_dataset/faulty_intervals.csv')
    parser.add_argument('--batch_registry', '-br', help="batch registry file path", type=str,
                        default='input_dataset/batch_registry.csv')
    parser.add_argument('--machine', '-m', help="machine id", type=str)
    parser.add_argument('--arepa_type', '-at', help="arepa type", type=str)
    parser.add_argument('--start_time', '-st', help="start time", type=str)
    parser.add_argument('--end_time', '-et', help="end time", type=str)
    parser.add_argument('--output', '-o', help="output file path", type=str)
    args = parser.parse_args()

    # Generate the training dataset
    final_dataset = generate_training_dataset(cooking_path=args.cooking_metrics,
                                              faulty_path=args.faulty_intervals,
                                              batch_path=args.batch_registry,
                                              machine_id=args.machine,
                                              arepa_type_name=args.arepa_type,
                                              start_time=args.start_time,
                                              end_time=args.end_time)

    # Save the final dataset to a CSV file
    try:
        final_dataset.to_csv(args.output, index=False)
        logger.info(f"Training dataset saved to {args.output}")
    except PermissionError as permission_error:
        logger.error(f"Error saving the file: {permission_error}")
        raise permission_error

    logger.info("Training dataset generation completed successfully")
