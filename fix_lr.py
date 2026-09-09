with open('run_uavdet_hpc.py', 'r') as f:
    code = f.read()

code = code.replace('_get_lr', '_lr')

with open('run_uavdet_hpc.py', 'w') as f:
    f.write(code)

print(">>> SUCCESS: Fixed _get_lr -> _lr in run_uavdet_hpc.py!")
