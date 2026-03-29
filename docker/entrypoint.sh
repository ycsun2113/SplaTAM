#!/bin/bash

sudo ldconfig

# Activate the splatam conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate splatam

exec /bin/bash