import subprocess
import os
os.environ["OMP_NUM_THREADS"] = "1"

def run_training():
    models = ['LSTM']
    angle_indices = ['2', '3']
    num_layers = ['6']
    for angle_idx in angle_indices:
        for model in models:
            for num_layer in num_layers:
                command = f"""
                CUDA_VISIBLE_DEVICES="0,1" \\
                torchrun \\
                    --nproc_per_node=2 \\
                    --nnodes=1 \\
                    --node_rank=0 \\
                main.py \\
                    --world_size=2 \\
                    --model_name {model} \\
                    --angle_idx {angle_idx} \\
                    --num_layers {num_layer} \\
                """
                print(f'|{model}-angle_{angle_idx}-num_layer:{num_layer}|: Beginning training')
                process = subprocess.Popen(command, shell=True)
                process.wait()  # 等待命令执行完成
                return_code = process.returncode
                if return_code != 0:
                    print(f'|{model}-angle_{angle_idx}-num_layer:{num_layer}|: Failed!')
                else:
                    print(f'|{model}-angle_{angle_idx}-num_layer:{num_layer}|: Successfully!')

if __name__ == "__main__":
    run_training()
