python run_eBPG.py --dataset enzymes --eps 0.1 0.01 0.001
python run_eBPG.py --dataset enzymes --noise_level 0.1 --eps 0.1 0.01 0.001
python run_eBPG.py --dataset proteins --eps 0.1 0.01 0.001
python run_eBPG.py --dataset proteins --noise_level 0.1 --eps 0.1 0.01 0.001
python run_eBPG.py --dataset reddit --eps 0.1 0.01 0.001
python run_eBPG.py --dataset reddit --noise_level 0.1 --eps 0.1 0.01 0.001
python generate_synthetic_data.py
python run_eBPG.py --dataset synthetic --eps 0.1 0.01 0.001
