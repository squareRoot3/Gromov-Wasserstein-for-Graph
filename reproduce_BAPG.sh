python run_BAPG.py --dataset enzymes --rho 0.5 0.2 0.1 0.05 0.01
python run_BAPG.py --dataset enzymes --noise_level 0.1 --rho 0.5 0.2 0.1 0.05 0.01
python run_BAPG.py --dataset proteins --rho 0.5 0.2 0.1 0.05 0.01
python run_BAPG.py --dataset proteins --noise_level 0.1 --rho 0.5 0.2 0.1 0.05 0.01
python run_BAPG.py --dataset reddit --use_gpu True --rho 0.5 0.2 0.1 0.05 0.01
python run_BAPG.py --dataset reddit --noise_level 0.1 --use_gpu True --rho 0.5 0.2 0.1 0.05 0.01
python generate_synthetic_data.py
python run_BAPG.py --dataset synthetic --use_gpu True --rho 0.5 0.2 0.1 0.05 0.01
