import pickle
import glob
metas = glob.glob('C:/MetaData/**/meta_data_pickle')
for meta in metas:
    with open(meta,'rb') as f:
        meta_dict = pickle.load(f)
        print(meta)
        if '218' in meta:
            continue
        print(len(meta_dict['s_chess_results']),len(meta_dict['b_chess_results']),len(meta_dict['m_chess_results']))