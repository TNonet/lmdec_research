"""

"""


def main():
    import os
    OMP_NUM_THREADS = '3'
    OPENBLAS_NUM_THREADS = '3'
    MKL_NUM_THREADS = '3'
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = OPENBLAS_NUM_THREADS  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = MKL_NUM_THREADS  # export MKL_NUM_THREADS=6

    import json
    import dask
    import dask.array as da
    import h5py
    import psutil
    import time
    import traceback

    from lmdec.decomp import PowerMethod

    data_directory = '/nfs/pool002/users/tnonet/'
    matrix_list = ['10K_40K.h5py', '20K_80K.h5py', '40K_160K.h5py', '80K_320K.h5py', '160K_640K.h5py']
    json_file_path = 'jan23_SVD_test1.json'

    matrix_list = [data_directory + i for i in matrix_list]

    for matrix_name in matrix_list:
        assert os.path.isfile(matrix_name)

    assert not os.path.isfile(json_file_path)

    logs = dict()
    sval_list = [10]
    max_iterations = 50
    buffer_list = [0, 5, 10, 20]
    tol = 1e-6
    num_runs = 1
    logs['sval_list'] = sval_list
    logs['num_runs'] = 1
    logs['max_iterations'] = max_iterations
    logs['buffer_list'] = buffer_list
    logs['tol'] = tol
    logs['cpus'] = psutil.cpu_count()
    logs['memory'] = psutil.virtual_memory()._asdict()
    logs['date'] = time.time()
    try:
        for matrix_name in matrix_list:
            size, _ = matrix_name.split('.')
            print(matrix_name)
            f = h5py.File(matrix_name, 'r')
            array = dask.array.from_array(f['/array'])
            array = array.rechunk({0: 'auto', 1: -1})
            for k in sval_list:
                for b in buffer_list:
                    for run in range(num_runs):
                        PM = PowerMethod(max_iter=max_iterations,
                                         k=k,
                                         buffer=b,
                                         scoring_tol=tol,
                                         )
                        _, _, _ = PM.svd(array)
                        logs[str((size, k, b, run))] = [PM.logs['tol'], PM.logs['start'], PM.logs['end']]

                    with open(json_file_path, 'w', encoding='utf-8') as f:
                        json.dump(logs, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(traceback.print_exc())
        pass


if __name__ == '__main__':
    main()
