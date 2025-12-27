import numpy as np
import pandas as pd
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
# Get the project root (one level up from features/)
PROJECT_ROOT = SCRIPT_DIR.parent
# Define data directory
DATA_DIR = PROJECT_ROOT / 'data'

# Define column names based on the schema
column_names = [
    'Unit ID (Engine Unit Identifier)',
    'Time (Cycles) (Operating Cycle Index)',
    'Altitude (Flight Altitude)',
    'Mach Number (Flight Mach Number)',
    'TRA (Throttle Resolver Angle)',
    'T2 (Fan Inlet Total Temperature)',
    'T24 (Low Pressure Compressor Outlet Total Temperature)',
    'T30 (High Pressure Compressor Outlet Total Temperature)',
    'T50 (Low Pressure Turbine Outlet Total Temperature)',
    'P2 (Fan Inlet Pressure)',
    'P15 (Bypass Duct Total Pressure)',
    'P30 (High Pressure Compressor Outlet Total Pressure)',
    'Ps30 (High Pressure Compressor Outlet Static Pressure)',
    'epr (Engine Pressure Ratio)',
    'Nf (Physical Fan Speed)',
    'Nc (Physical Core Speed)',
    'NRf (Corrected Fan Speed)',
    'NRc (Corrected Core Speed)',
    'Nf_dmd (Demanded Fan Speed)',
    'PCNfR_dmd (Demanded Corrected Fan Speed)',
    'phi (Fuel Flow to Ps30 Ratio)',
    'BPR (Bypass Ratio)',
    'farB (Burner Fuel Air Ratio)',
    'htBleed (Bleed Enthalpy)',
    'W31 (High Pressure Turbine Coolant Bleed Flow)',
    'W32 (Low Pressure Turbine Coolant Bleed Flow)'
]

# Dataset 1
dataset_train_1 = pd.read_csv(
    DATA_DIR / 'train_FD001.txt',
    delim_whitespace=True,
    header=None,
    names=column_names
)

dataset_test_1 = pd.read_csv(
    DATA_DIR / 'test_FD001.txt',
    delim_whitespace=True,
    header=None,
    names=column_names
)

# Dataset 2
dataset_train_2 = pd.read_csv(
    DATA_DIR / 'train_FD002.txt',
    delim_whitespace=True,
    header=None,
    names=column_names
)

dataset_test_2 = pd.read_csv(
    DATA_DIR / 'test_FD002.txt',
    delim_whitespace=True,
    header=None,
    names=column_names
)

# Dataset 3
dataset_train_3 = pd.read_csv(
    DATA_DIR / 'train_FD003.txt',
    delim_whitespace=True,
    header=None,
    names=column_names
)

dataset_test_3 = pd.read_csv(
    DATA_DIR / 'test_FD003.txt',
    delim_whitespace=True,
    header=None,
    names=column_names
)

# Dataset 4
dataset_train_4 = pd.read_csv(
    DATA_DIR / 'train_FD004.txt',
    delim_whitespace=True,
    header=None,
    names=column_names
)

dataset_test_4 = pd.read_csv(
    DATA_DIR / 'test_FD004.txt',
    delim_whitespace=True,
    header=None,
    names=column_names
)

print(dataset_train_1.head())