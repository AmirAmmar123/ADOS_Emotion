from Files import * 
import numpy as np
import pandas as pd
from os import mkdir 
from os.path import exists
import matplotlib.pyplot as plt
AS_TYPE = {'record':'int32','Type':'string','Gender':'string'}
RECORD2TYPE = pd.read_excel('record2type.xlsx',engine='openpyxl').astype(AS_TYPE).set_index('record')
relevantKps = [0,1,2,3,4,5,6,7,17,15,16,18]
kps_cols =[]
for kp in relevantKps:
    kps_cols.append(str(kp)+"-1dist-normed")
    # kps_cols.append(str(kp)+"_y")

data = pd.read_csv(get_file())
for index,row in data.iterrows():
    record = row['record']
    type = RECORD2TYPE.at[record,'Type']
    gender = RECORD2TYPE.at[record,'Gender']
    data.at[index,'Type'] = type
    data.at[index,'Gender'] = gender
# data.set_index(['record','task'],inplace=True)
print(data)

all = data[kps_cols+['Type','Gender']]
print(all)
if not exists('hands_histograms/'):
    mkdir('hands_histograms')
# all.drop('record',axis=1,inplace=True)
MALE_MASK = all['Gender'] == 'M'
FEMALE_MASK = all['Gender'] == 'F'
ASD_mask = all['Type'] == 'ASD'
NT_mask = all['Type'] == 'NT'
male = all[MALE_MASK]
female = all[FEMALE_MASK]
M_ASD_mask = male['Type'] == 'ASD'
M_NT_mask = male['Type'] == 'NT'
F_ASD_mask = female['Type'] == 'ASD'
F_NT_mask = female['Type'] == 'NT'
for col in all.columns:
    if col == 'Type' or col =='Gender':
        continue
    all[NT_mask][col].plot.hist(bins=20,alpha=0.5,title='NT')
    all[ASD_mask][col].plot.hist(bins=20,alpha=0.5,title='ASD')
    mean_nt = all[NT_mask][col].mean()
    mean_asd = all[ASD_mask][col].mean()
    plt.text(mean_nt,plt.ylim()[1]/2,'mean NT: '+"{:.3f}".format(mean_nt),rotation=90)
    plt.text(mean_asd,plt.ylim()[1]/2,'mean ASD: '+"{:.3f}".format(mean_asd),rotation=90)
    plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
    plt.legend(['ASD','NT'])
    counts = all["Type"].value_counts()
    plt.title(col + "(ASD: "+str(counts['ASD'])+", NT: "+str(counts['NT'])+")")
    if not exists('hands_histograms/type'):
        mkdir('hands_histograms/type')
    plt.savefig('hands_histograms/type/'+col)
    plt.close()
for col in all.columns:
    if col == 'Type' or col =='Gender':
        continue
    male[col].plot.hist(bins=20,alpha=0.5,title='M')
    female[col].plot.hist(bins=20,alpha=0.5,title='F')
    mean_nt =  male[col].mean()
    mean_asd = female[col].mean()
    plt.text(mean_nt,plt.ylim()[1]/2,'mean M: '+"{:.3f}".format(mean_nt),rotation=90)
    plt.text(mean_asd,plt.ylim()[1]/2,'mean F: '+"{:.3f}".format(mean_asd),rotation=90)
    plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
    plt.legend(['M','F'])
    counts = all["Gender"].value_counts()
    plt.title(col + "(M: "+str(counts['M'])+", F: "+str(counts['F'])+")")
    plt.title(col)
    if not exists('hands_histograms/gender'):
        mkdir('hands_histograms/gender')
    plt.savefig('hands_histograms/gender/'+col)
    plt.close()
for col in male.columns:
    if col == 'Type' or col =='Gender':
        continue
    male[M_NT_mask][col].plot.hist(bins=20,alpha=0.5,title='NT')
    male[M_ASD_mask][col].plot.hist(bins=20,alpha=0.5,title='ASD')
    mean_nt =  male[M_NT_mask][col].mean()
    mean_asd = male[M_ASD_mask][col].mean()
    plt.text(mean_nt,plt.ylim()[1]/2,'mean NT: '+"{:.3f}".format(mean_nt),rotation=90)
    plt.text(mean_asd,plt.ylim()[1]/2,'mean ASD: '+"{:.3f}".format(mean_asd),rotation=90)
    plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
    plt.legend(['ASD','NT'])
    counts = male["Type"].value_counts()
    plt.title(col +" Male "+"(ASD: "+str(counts['ASD'])+", NT: "+str(counts['NT'])+")")
    if not exists('hands_histograms/male'):
        mkdir('hands_histograms/male')
    plt.savefig('hands_histograms/male/male_'+col)
    plt.close()
for col in female.columns:
    if col == 'Type' or col =='Gender':
        continue
    female[F_NT_mask][col].plot.hist(bins=20,alpha=0.5,title='NT')
    female[F_ASD_mask][col].plot.hist(bins=20,alpha=0.5,title='ASD')
    mean_nt =  female[F_NT_mask][col].mean()
    mean_asd = female[F_ASD_mask][col].mean()
    plt.text(mean_nt,plt.ylim()[1]/2,'mean NT: '+"{:.3f}".format(mean_nt),rotation=90)
    plt.text(mean_asd,plt.ylim()[1]/2,'mean ASD: '+"{:.3f}".format(mean_asd),rotation=90)
    plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
    plt.legend(['ASD','NT'])
    counts = female["Type"].value_counts()
    plt.title(col +" Female "+"(ASD: "+str(counts['ASD'])+", NT: "+str(counts['NT'])+")")
    # plt.title(col + " Female")
    if not exists('hands_histograms/female'):
        mkdir('hands_histograms/female')
    plt.savefig('hands_histograms/female/female_'+col)
    plt.close()