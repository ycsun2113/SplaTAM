#!/bin/bash

# run sub20 all
python3 scripts/splatam.py configs/iphone/stray_scan.py

python3 scripts/post_splatam_opt.py configs/iphone/stray_scan_opt.py

python3 scripts/export_ply.py configs/iphone/stray_scan_opt.py


# run sub15 all
python3 scripts/splatam.py configs/iphone/stray_scan_15.py

python3 scripts/post_splatam_opt.py configs/iphone/stray_scan_opt_15.py

python3 scripts/export_ply.py configs/iphone/stray_scan_opt_15.py


# run r3d scan sub20 all
# python3 scripts/splatam.py configs/iphone/r3d_scan_new.py

# python3 scripts/post_splatam_opt.py configs/iphone/r3d_post_opt.py

# python3 scripts/export_ply.py configs/iphone/r3d_post_opt.py

# python3 scripts/splatam.py configs/iphone/r3d_scan_new.py

# python3 scripts/post_splatam_opt.py configs/iphone/r3d_scan_opt.py

# python3 scripts/export_ply.py configs/iphone/r3d_scan_opt.py

