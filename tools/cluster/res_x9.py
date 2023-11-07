"""saving all trajectories in pandas HDF format for easier access on machine
with less memory"""

import pandas as pd
import argparse
from tqdm import tqdm
from copy import deepcopy


def resave_trajectories(path_in,
                        path_out):
    """load dataframe from path_in and resave trajectories in HDF store at
    path_out. Name index levels according to index_levels
    """
    print('loading data...')
    data = pd.read_pickle(path_in)
    print('done.')
    index = pd.Index(range(len(data.columns.values)))
    index.name = 'run_id'
    data.columns = index
    data = data.stack(index.name)

    saved_indices = []

    data_tmp = data.copy()
    index_levels = deepcopy(data.index.names)
    with pd.HDFStore(path_out) as store:
        app = False
        with tqdm(total=len(data)) as pbar:
            for i, dfl in data_tmp.iteritems():
                pbar.update(1)
                if i[0] < 90 or i[1] < 90:
                    continue
                if 1 not in saved_indices:
                    for val, name in zip(i, index_levels):
                        dfl[name] = val
                    dfl.set_index(index_levels + ['time'], inplace=True)

                    if app:
                        try:
                            store.append('d1', dfl.astype(float))
                        except:
                            print(f'{i} failed')
                            print(dfl)
                            raise
                    else:
                        print(dfl.columns)
                        store.put('d1',
                                  dfl.astype(float),
                                  append=app,
                                  format='table',
                                  data_columns=True)
                    app = True
                    saved_indices.append(i)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--path_in", required=True, help="path to input file")
    ap.add_argument("-o", "--path_out", required=True, help="path to output file")
    args = vars(ap.parse_args())
    resave_trajectories(path_in=args['path_in'], path_out=args['path_out'])
