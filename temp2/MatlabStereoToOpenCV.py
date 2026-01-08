import pickle
import glob
import matlab.engine
import pickle
import glob
import os
import numpy as np
from CheckCreate import check_create
metas = glob.glob('C:/MetaData2/*')
print(metas)
eng = matlab.engine.start_matlab()
for meta in metas:
    # if meta.split('\\')[-1]!='218':
    #     continue
    print(meta)
    if not (os.path.exists(meta+'/s_b_final_stereo.mat') and os.path.exists(meta+'/m_b_final_stereo.mat')):
        print('NO ', meta)
        continue
    s_b_final_stereo =  eng.load(meta+'/s_b_final_stereo.mat')
    m_b_final_stereo =  eng.load(meta+'/m_b_final_stereo.mat')
    eng.workspace['sbs'] = s_b_final_stereo['s_b_final_stereo'][0]
    eng.workspace['mbs'] = m_b_final_stereo['m_b_final_stereo'][0]
    print(eng.eval('sbs.MeanReprojectionError'))
    sbs = eng.stereoParametersToOpenCV(eng.workspace['sbs'],nargout=6)
    mbs = eng.stereoParametersToOpenCV(eng.workspace['mbs'],nargout=6)
    opencv_sbs = [] 
    opencv_mbs = []
    for matItem in sbs:
        opencv_sbs.append(np.array([item.toarray() for item in matItem]))
        # print(opencv_sbs)
        continue
    for matItem in mbs:
        opencv_mbs.append(np.array([item.toarray() for item in matItem]))
    check_create(meta+'/openCV')
    with open(meta+'/openCV/s_b_stereoParams','wb') as f:
        pickle.dump(opencv_sbs,f)
    with open(meta+'/openCV/m_b_stereoParams','wb') as f:
        pickle.dump(opencv_mbs,f)
    if os.path.exists(meta+'/s_m_final_stereo.mat'):
        s_b_final_stereo =  eng.load(meta+'/s_m_final_stereo.mat')
        eng.workspace['sms'] = s_b_final_stereo['s_m_final_stereo'][0]
        print('SM',eng.eval('sms.MeanReprojectionError'))
        sms = eng.stereoParametersToOpenCV(eng.workspace['sms'],nargout=6)
        opencv_sms = []
        for matItem in sms:
            opencv_sms.append(np.array([item.toarray() for item in matItem]))
        with open(meta+'/openCV/s_m_stereoParams','wb') as f:
            pickle.dump(opencv_sms,f)
