import json, re, unicodedata
from io import StringIO
from urllib.request import urlopen
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

U24='https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/players_raw.csv'
UMAP='https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/2025-2026/By%20Gameweek/GW10/players.csv'
UGW10='https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/2025-2026/By%20Gameweek/GW10/playerstats.csv'

def read_url(u):
    return pd.read_csv(StringIO(urlopen(u).read().decode('utf-8-sig')))

def norm(s):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+','',s)

def cstat(x,y,kind='s'):
    x=pd.Series(x,dtype=float); y=pd.Series(y,dtype=float); m=x.notna()&y.notna(); x=x[m]; y=y[m]
    if len(x)<4 or x.nunique()<2 or y.nunique()<2: return {'r':None,'p':None,'n':int(len(x))}
    r,p=(spearmanr(x,y) if kind=='s' else pearsonr(x,y)); return {'r':float(r),'p':float(p),'n':int(len(x))}

def cv_r2(X,y):
    k=min(10,max(3,len(y)//20)); cv=KFold(k,shuffle=True,random_state=42)
    z=cross_val_score(LinearRegression(),X,y,cv=cv,scoring='r2')
    return {'mean':float(np.mean(z)),'sd':float(np.std(z)),'folds':int(k)}

def segment(d,label):
    d=d.copy(); y=d.f10_points.astype(float)
    out={'segment':label,'n':int(len(d)),'correlations':{}}
    for key,col in [('full','points_2425'),('early','gw1_28'),('late','gw29_38'),('acceleration','acceleration')]:
        out['correlations'][key]={'spearman':cstat(d[col],y,'s'),'pearson':cstat(d[col],y,'p')}
    m0=sm.OLS(y,sm.add_constant(d[['early_rate']])).fit()
    m1=sm.OLS(y,sm.add_constant(d[['early_rate','late_rate']])).fit()
    out['regression']={'early_only_r2':float(m0.rsquared),'early_plus_late_r2':float(m1.rsquared),'incremental_r2':float(m1.rsquared-m0.rsquared),'late_coefficient':float(m1.params['late_rate']),'late_p':float(m1.pvalues['late_rate']),'early_coefficient':float(m1.params['early_rate']),'early_p':float(m1.pvalues['early_rate'])}
    out['cv_r2']={'early_only':cv_r2(d[['early_rate']].values,y.values),'early_plus_late':cv_r2(d[['early_rate','late_rate']].values,y.values)}
    lm=LinearRegression().fit(d[['early_rate']],d.late_rate); d['late_residual']=d.late_rate-lm.predict(d[['early_rate']])
    out['residual_late']={'spearman':cstat(d.late_residual,y,'s'),'pearson':cstat(d.late_residual,y,'p')}
    ql=float(d.gw29_38.quantile(.75)); qf=float(d.f10_points.quantile(.75)); qfull=float(d.points_2425.quantile(.75))
    d['top_late']=d.gw29_38>=ql; d['top_f10']=d.f10_points>=qf; d['top_full']=d.points_2425>=qfull
    base=float(d.top_f10.mean()); hl=float(d.loc[d.top_late,'top_f10'].mean()); hf=float(d.loc[d.top_full,'top_f10'].mean())
    out['quartiles']={'late_threshold':ql,'f10_threshold':qf,'full_threshold':qfull,'base_rate':base,'top_late_count':int(d.top_late.sum()),'top_late_and_f10':int((d.top_late&d.top_f10).sum()),'top_late_hit_rate':hl,'top_late_lift':hl/base if base else None,'top_full_count':int(d.top_full.sum()),'top_full_and_f10':int((d.top_full&d.top_f10).sum()),'top_full_hit_rate':hf,'top_full_lift':hf/base if base else None,'mean_f10_top_late':float(d.loc[d.top_late,'f10_points'].mean()),'mean_f10_others':float(d.loc[~d.top_late,'f10_points'].mean())}
    return out,d

old=pd.read_csv('fpl_bridge_old.csv')
p24=read_url(U24)
mp=read_url(UMAP)
gw=read_url(UGW10)
p24['full_name']=(p24.first_name.fillna('')+' '+p24.second_name.fillna('')).str.strip()
p24['nfull']=p24.full_name.map(norm); p24['nweb']=p24.web_name.map(norm)
old['n']=old.player.map(norm)
fullmap=dict(zip(p24.nfull,p24.code)); webmap={k:v.iloc[0].code for k,v in p24.groupby('nweb') if len(v)==1}
# Explicit aliases cover FPL long-name conventions and name-order anomalies.
alias={
 'mitomakaoru':'mitoma','matheussantoscarneirodacunha':'cunha','brunoborgesfernandes':'bfernandes','brunoguimaraesrodriguezmoura':'brunog','davidayamartin':'raya','robertsanchez':'sanchez','marccucurellasaseta':'cucurella','murillosantiagocostadossantos':'murillo','joaopedrojunqueiradejesus':'joaopedro','franciscoevanilsondelimabarbosa':'evanilson','gabrieldossantosmagalhaes':'gabriel','jurriëntimber':'jtimber','alissonramsesbecker':'alisson','edersonsantanademoraes':'ederson','emilianomartinezromero':'martinez','bernardoveigadecarvalhoesilva':'bernardo','dominicsolankemitchell':'solanke','saviosavinhomoreiradeoliveira':'savinho','rubengatoalvesdias':'dias','diogoteixeiradasilva':'jota','norbertoberciquegomesbetuncal':'beto','carloshenriquecasimiro':'casemiro','joaopedroferreirasilva':'jpedro','martindubravka':'dubravka','gabrielfernandodejesus':'gjesus','saalukic':'lukic','lukaszfabianski':'fabianski'
}
def code_for(n):
    if n in fullmap:return fullmap[n]
    if n in webmap:return webmap[n]
    a=alias.get(n)
    if a and a in webmap:return webmap[a]
    return np.nan
old['code']=old.n.map(code_for)
mp=mp.rename(columns={'player_code':'code','player_id':'id25'})
gw=gw.rename(columns={'id':'id25','total_points':'f10_points','minutes':'f10_minutes'})
b=old.merge(mp[['code','id25','web_name']],on='code',how='left').merge(gw[['id25','f10_points','f10_minutes']],on='id25',how='left')
b=b[(b.points_2425>=30)&b.f10_points.notna()].copy()
b['early_rate']=b.gw1_28/28; b['late_rate']=b.gw29_38/10; b['acceleration']=b.late_rate-b.early_rate
b['group']=np.where(b.position.isin(['MID','FWD']),'MID/FWD','DEF/GK')
a,ad=segment(b,'All'); mf,mfd=segment(b[b.group=='MID/FWD'],'MID/FWD'); dg,dgd=segment(b[b.group=='DEF/GK'],'DEF/GK')
avail=b[b.f10_minutes>=450].copy(); av, avd=segment(avail,'450+ minutes')
d=ad
cols=['player','web_name','position','points_2425','gw1_28','gw29_38','acceleration','f10_points','f10_minutes']
def rec(x,n=15):return x[cols].head(n).replace({np.nan:None}).to_dict('records')
good=d[d.top_late&d.top_f10].sort_values(['f10_points','gw29_38'],ascending=False)
fp=d[d.top_late&~d.top_f10].sort_values(['gw29_38','f10_points'],ascending=[False,True])
fn=d[~d.top_late&d.top_f10].sort_values(['f10_points','gw29_38'],ascending=[False,True])
unmatched=old[(old.points_2425>=30)&~old.code.isin(b.code)].copy()
r={'sources':{'old_input':'fpl_bridge_old.csv','players_2425':U24,'map_2526':UMAP,'gw10_2526':UGW10},'sample':{'eligible_2425':int((old.points_2425>=30).sum()),'matched':int(len(b)),'matched_450plus_minutes':int(len(avail)),'unmatched_or_not_in_2526':int((old.points_2425>=30).sum()-len(b))},'segments':{'all':a,'mid_fwd':mf,'def_gk':dg,'minutes_450_plus':av},'examples':{'success':rec(good),'false_positive':rec(fp),'false_negative':rec(fn)},'unmatched_high_points':unmatched.sort_values('points_2425',ascending=False)[['player','position','points_2425','code']].head(40).replace({np.nan:None}).to_dict('records')}
open('bridge_results.json','w',encoding='utf-8').write(json.dumps(r,indent=2,ensure_ascii=False))
b[cols+['code','id25','group']].sort_values('gw29_38',ascending=False).to_csv('bridge_panel.csv',index=False)
print(json.dumps(r['sample']))
