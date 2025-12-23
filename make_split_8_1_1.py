import os, h5py, numpy as np

DS_PATH = os.path.expanduser('~/Downloads/Data_Collecting/bc_dataset_from616.hdf5')
SEED = 201
OVERWRITE = True   # set False to avoid touching existing masks

def main():
    with h5py.File(DS_PATH, 'r+') as f:
        data = f['data']
        # collect demo groups that actually have samples
        demos = []
        for name, grp in data.items():
            if isinstance(grp, h5py.Group):
                n = int(grp.attrs.get('num_samples', 0))
                if n > 0:
                    demos.append((name, n))
        if not demos:
            raise RuntimeError("No non-empty demos found in /data")

        demos.sort(key=lambda x: x[0])  # deterministic before shuffle
        rng = np.random.default_rng(SEED)
        rng.shuffle(demos)

        N = len(demos)
        n_train = (N * 8) // 10
        n_val   = (N * 1) // 10
        n_test  = N - n_train - n_val

        train = [d[0] for d in demos[:n_train]]
        val   = [d[0] for d in demos[n_train:n_train+n_val]]
        test  = [d[0] for d in demos[n_train+n_val:]]

        # summary by number of transitions (may be slightly different % than by demos)
        def count_trans(keys): return int(sum(int(data[k].attrs['num_samples']) for k in keys))
        tot_trans = count_trans([d[0] for d in demos])
        t_train, t_val, t_test = map(count_trans, (train, val, test))

        print(f"Demos: total={N}  train={len(train)}  val={len(val)}  test={len(test)}")
        print(f"Transitions: total={tot_trans} | "
              f"train={t_train} ({t_train/tot_trans:.1%})  "
              f"val={t_val} ({t_val/tot_trans:.1%})  "
              f"test={t_test} ({t_test/tot_trans:.1%})")

        # write mask datasets
        m = f.require_group('mask')
        dt = h5py.string_dtype(encoding='utf-8')

        def write_mask(key, values):
            if key in m:
                if not OVERWRITE:
                    raise RuntimeError(f"mask/{key} already exists; set OVERWRITE=True to replace.")
                del m[key]
            m.create_dataset(key, data=np.array(values, dtype=dt))
            print(f"mask/{key}: {len(values)} demos")

        write_mask('train', train)
        write_mask('valid', val)   # robomimic convention uses 'valid'
        write_mask('test',  test)

        # ---- ALSO save selected demo names to ~/Downloads ----
        downloads_dir = os.path.expanduser('~/Downloads')
        os.makedirs(downloads_dir, exist_ok=True)

        # 1) Single CSV with split label and num_samples
        csv_path = os.path.join(downloads_dir, f'bc_split_8_1_1_seed{SEED}.csv')
        with open(csv_path, 'w', encoding='utf-8') as fout:
            fout.write('demo,split,num_samples\n')
            for split_name, keys in (('train', train), ('valid', val), ('test', test)):
                for k in keys:
                    n = int(data[k].attrs['num_samples'])
                    fout.write(f'{k},{split_name},{n}\n')
        print(f"Saved split CSV: {csv_path}")

        # 2) One TXT per split (just the demo names, one per line)
        for split_name, keys in (('train', train), ('valid', val), ('test', test)):
            txt_path = os.path.join(downloads_dir, f'bc_{split_name}_demos_seed{SEED}.txt')
            with open(txt_path, 'w', encoding='utf-8') as fout:
                fout.write('\n'.join(keys) + '\n')
            print(f"Saved {split_name} list: {txt_path}")


if __name__ == "__main__":
    main()
