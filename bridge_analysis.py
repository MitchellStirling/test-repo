import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import statsmodels.api as sm

OUT_JSON = Path('bridge_results.json')
OUT_CSV = Path('bridge_panel.csv')
URL_PLAYERS_2425='https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/players_raw.csv'
URL_GWS_2425='https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv'
URL_MAP_2526='https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/2025-2026/By%20Gameweek/GW10/players.csv'
URL_GW10_2526='https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/2025-2026/By%20Gameweek/GW10/playerstats.csv'

def cor(f,x,y):
 x=pd.Series(x,dtype=float);y=pd.Series(y,dtype=float);m=x.notna()&y.notna();x=x[m];y=y[m]
 if len(x)<4 or x.nunique()<2 or y.nunique()<2:return {'r':None,'p':None,'n':int(len(x))}
 r,p=f(x,y);return {'r':float(r),'p':float(p),'n':int(len(x))}
def cv(X,y):
 k=min(10,max(3,len(y)//20));s=cross_val_score(LinearRegression(),X,y,cv=KFold(k,shuffle=True,random_state=42),scoring='r2');return {'mean':float(s.mean()),'sd':float(s.std()),'folds':k}
def stats(d,label):
 d=d.copy();y=d.f10_points.astype(float)
 out={'segment':label,'n':int(len(d)),'correlations':{}}
 for key,col in [('full','points_2425'),('early','gw1_28'),('late','gw29_38'),('acceleration','acceleration')]:out['correlations'][key]={'spearman':cor(spearmanr,d[col],y),'pearson':cor(pearsonr,d[col],y)}
 m0=sm.OLS(y,sm.add_constant(d[['early_rate']])).fit();m1=sm.OLS(y,sm.add_constant(d[['early_rate','late_rate']])).fit()
 out['regression']={'early_only_r2':float(m0.rsquared),'early_plus_late_r2':float(m1.rsquared),'incremental_r2':float(m1.rsquared-m0.rsquared),'late_coefficient':float(m1.params.late_rate),'late_p':float(m1.pvalues.late_rate),'early_coefficient':float(m1.params.early_rate),'early_p':float(m1.pvalues.early_rate)}
 out['cv_r2']={'early_only':cv(d[['early_rate']].values,y.values),'early_plus_late':cv(d[['early_rate','late_rate']].values,y.values)}
 lm=LinearRegression().fit(d[['early_rate']],d.late_rate);d['late_residual']=d.late_rate-lm.predict(d[['early_rate']]);out['residual_late']={'spearman':cor(spearmanr,d.late_residual,y),'pearson':cor(pearsonr,d.late_residual,y)}
 ql=float(d.gw29_38.quantile(.75));qf=float(d.f10_points.quantile(.75));qs=float(d.points_2425.quantile(.75));d['top_late']=d.gw29_38>=ql;d['top_f10']=d.f10_points>=qf;d['top_full']=d.points_2425>=qs;base=float(d.top_f10.mean());hl=float(d.loc[d.top_late,'top_f10'].mean());hf=float(d.loc[d.top_full,'top_f10'].mean())
 out['quartiles']={'late_threshold':ql,'f10_threshold':qf,'full_threshold':qs,'base_rate':base,'top_late_count':int(d.top_late.sum()),'top_late_and_f10':int((d.top_late&d.top_f10).sum()),'top_late_hit_rate':hl,'top_late_lift':hl/base,'top_full_count':int(d.top_full.sum()),'top_full_and_f10':int((d.top_full&d.top_f10).sum()),'top_full_hit_rate':hf,'top_full_lift':hf/base,'mean_f10_top_late':float(d.loc[d.top_late,'f10_points'].mean()),'mean_f10_others':float(d.loc[~d.top_late,'f10_points'].mean())}
 return out,d
p=pd.read_csv(URL_PLAYERS_2425);g=pd.read_csv(URL_GWS_2425);mp=pd.read_csv(URL_MAP_2526);w=pd.read_csv(URL_GW10_2526)
gwc=next(c for c in ['GW','round','event'] if c in g.columns);wk=g.groupby(['element',gwc],as_index=False).total_points.sum();e=wk[wk[gwc]<=28].groupby('element').total_points.sum().rename('gw1_28');l=wk[wk[gwc]>=29].groupby('element').total_points.sum().rename('gw29_38')
s=p[['id','code','first_name','second_name','web_name','element_type','total_points']].rename(columns={'id':'element','total_points':'points_2425'}).join(e,on='element').join(l,on='element').fillna({'gw1_28':0,'gw29_38':0});s['player']=s.first_name.fillna('')+' '+s.second_name.fillna('');s['position']=s.element_type.map({1:'GK',2:'DEF',3:'MID',4:'FWD'});s['early_rate']=s.gw1_28/28;s['late_rate']=s.gw29_38/10;s['acceleration']=s.late_rate-s.early_rate;s=s[s.points_2425>=30]
if 'player_code' not in mp:mp=mp.rename(columns={'code':'player_code'})
if 'player_id' not in mp:mp=mp.rename(columns={'id':'player_id'})
w=w.rename(columns={'id':'player_id','total_points':'f10_points','minutes':'f10_minutes'})
b=s.merge(mp[['player_code','player_id']],left_on='code',right_on='player_code',how='left').merge(w[['player_id','f10_points','f10_minutes']],on='player_id',how='left');b=b[b.f10_points.notna()].copy();b['group']=np.where(b.position.isin(['MID','FWD']),'MID/FWD','DEF/GK')
a,ad=stats(b,'All');mf,mfd=stats(b[b.group=='MID/FWD'],'MID/FWD');dg,dgd=stats(b[b.group=='DEF/GK'],'DEF/GK')
d=ad;good=d[d.top_late&d.top_f10].sort_values(['f10_points','gw29_38'],ascending=False);fp=d[d.top_late&~d.top_f10].sort_values(['gw29_38','f10_points'],ascending=[False,True]);fn=d[~d.top_late&d.top_f10].sort_values(['f10_points','gw29_38'],ascending=[False,True]);cols=['player','web_name','position','points_2425','gw1_28','gw29_38','acceleration','f10_points','f10_minutes'];rec=lambda x,n=15:x[cols].head(n).replace({np.nan:None}).to_dict('records')
r={'sources':{'players_2425':URL_PLAYERS_2425,'weekly_2425':URL_GWS_2425,'map_2526':URL_MAP_2526,'gw10_2526':URL_GW10_2526},'sample':{'eligible_2425':int(len(s)),'matched':int(len(b)),'excluded':int(len(s)-len(b))},'segments':{'all':a,'mid_fwd':mf,'def_gk':dg},'examples':{'success':rec(good),'false_positive':rec(fp),'false_negative':rec(fn)}}
OUT_JSON.write_text(json.dumps(r,indent=2,ensure_ascii=False));b[cols+['code','player_id','group']].sort_values('gw29_38',ascending=False).to_csv(OUT_CSV,index=False)
print(json.dumps(r['sample']))
