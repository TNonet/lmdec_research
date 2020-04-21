
def main():
    import os
    OMP_NUM_THREADS = '1'
    OPENBLAS_NUM_THREADS = '1'
    MKL_NUM_THREADS = '1'
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = OPENBLAS_NUM_THREADS  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = MKL_NUM_THREADS  # export MKL_NUM_THREADS=6

    import json
    import zarr
    import dask.array as da
    from dask.distributed import Client
    import time
    import traceback

    from lmdec.decomp import PowerMethod

    data_directory = '/nfs/pool002/users/tnonet/SNP_Zarr'
    #data_directory = '/Users/tnonet/Documents/SNP_matrices'
    matrix = '200K_400K'
    matrix_path = os.path.join(data_directory, matrix + '.zarr')
    json_file_path = '_'.join(['March1_2', 'PM_test', matrix]) + '.json'

    assert os.path.isdir(matrix_path)

    assert not os.path.isfile(json_file_path)

    logs = dict()
    k = 10
    max_iterations = 200
    time_limit = 6400
    buffer = 10
    tol = 1e-6
    num_runs = 1
    p = 1
    worker_list = [2, 4, 8]
    memory_list = ['100GB', '200GB', '400GB']
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
    logs['worker'] = worker_list
    logs['memory'] = memory_list
    try:
        root = zarr.open(matrix_path, mode='r')
        array = da.from_zarr(root)
        for run in range(num_runs):
            for work in worker_list:
                for mem in memory_list:
                    client = Client(n_workers=work,
                                    threads_per_worker=1,
                                    memory_limit=mem)
                    PM = PowerMethod(max_iter=max_iterations,
                                     k=k,
                                     buffer=buffer,
                                     p=p,
                                     tol=tol,
                                     scoring_method=score,
                                     time_limit=time_limit,
                                     track_metrics=True)
                    _, _, _ = PM.svd(array)
                    client.close()
                    logs[str((work, mem, run))] = [str(PM.metrics), str(PM.time), str(PM.times)]

                    with open(json_file_path, 'w', encoding='utf-8') as f:
                        json.dump(logs, f, ensure_ascii=False, indent=4)


    except Exception:
        traceback.print_exc()
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
