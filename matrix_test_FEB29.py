import sys

def main(memory, worker):
    import os
    OMP_NUM_THREADS = '1'
    OPENBLAS_NUM_THREADS = '1'
    MKL_NUM_THREADS = '1'
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = OPENBLAS_NUM_THREADS  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = MKL_NUM_THREADS  # export MKL_NUM_THREADS=6

    import json
    import dask
    import dask.array as da
    import h5py
    import time
    import traceback

    from lmdec.decomp import PowerMethod

    #data_directory = '/Users/tnonet/Documents/FlashPCATests/run_directory/matricies'
    data_directory = '/nfs/pool002/users/tnonet/SNP_Uncompressed'
    matrix = '160K_640K'
    matrix_path = os.path.join(data_directory, matrix + '.h5py')
    json_file_path = '_'.join(['Feb29', 'PM_test', matrix, worker, memory]) + '.json'


    assert os.path.isfile(matrix_path)

    assert not os.path.isfile(json_file_path)

    logs = dict()
    k = 10
    max_iterations = 200
    time_limit = 12000
    buffer = 10
    tol = 1e-6
    num_runs = 1
    p = 1
    score = 'rmse'
    logs['k'] = k
    logs['p'] = p
    logs['num_runs'] = num_runs
    logs['max_iterations'] = max_iterations
    logs['scoring'] = score
    logs['b'] = buffer
    logs['time_limit'] = time_limit
    logs['tol'] = tol
    logs['date'] = time.time()
    logs['worker'] = worker
    logs['memory'] = memory
    try:
        f = h5py.File(matrix_path, 'r')
        array = dask.array.from_array(f['/array'])
        for run in range(num_runs):
            PM = PowerMethod(max_iter=max_iterations,
                             k=k,
                             buffer=buffer,
                             p=p,
                             tol=tol,
                             scoring_method=score,
                             time_limit=time_limit,
                             track_metrics=True)
            _, _, _ = PM.svd(array)
            logs[str((worker, run))] = [str(PM.metrics), str(PM.time), str(PM.times)]

            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=4)

    except Exception:
        traceback.print_exc()
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':

    main(sys.argv[1], sys.argv[2])
