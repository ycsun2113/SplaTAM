#!/bin/bash

python3 scripts/splatam.py configs/iphone/r3d_scan_new.py

python3 scripts/post_splatam_opt.py configs/iphone/r3d_post_opt.py