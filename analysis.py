"""
Heart Disease Analysis & Prediction
Data: UCI Heart Disease (Cleveland, Hungary, Switzerland, Long Beach VA) - 920 patients
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, classification_report,
                             precision_recall_curve, average_precision_score)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

OUT = '/home/claude/work/figs'
import os; os.makedirs(OUT, exist_ok=True)

# ---- brand palette ----
C = {'primary':'#B02A37','secondary':'#1F4E79','accent':'#E8A33D','ok':'#2E7D5B',
     'grey':'#6B7280','light':'#F4F1EC','dark':'#1A1A1A'}
plt.rcParams.update({
    'figure.dpi':130,'savefig.dpi':130,'font.family':'DejaVu Sans','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
    'grid.alpha':0.25,'grid.linestyle':'-','grid.linewidth':0.6,
    'axes.edgecolor':'#CCCCCC','figure.facecolor':'white','axes.facecolor':'white'
})

# ============================================================
# 1. LOAD  — merge all four study sites
# ============================================================
COLS = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
        'exang','oldpeak','slope','ca','thal','num']
SITES = {'Cleveland':'processed.cleveland.data','Hungary':'processed.hungarian.data',
         'Switzerland':'processed.switzerland.data','Long Beach VA':'processed.va.data'}

frames = []
for site, f in SITES.items():
    d = pd.read_csv(f'/home/claude/work/uci_ds/{f}', names=COLS, na_values='?')
    d['site'] = site
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

# Switzerland records cholesterol as 0 = not measured
df.loc[df.chol == 0, 'chol'] = np.nan
df.loc[df.trestbps == 0, 'trestbps'] = np.nan

df['target'] = (df.num > 0).astype(int)

LABELS = {
 'age':'Age','sex':'Sex','cp':'Chest pain type','trestbps':'Resting BP (mm Hg)',
 'chol':'Cholesterol (mg/dl)','fbs':'Fasting blood sugar >120','restecg':'Resting ECG',
 'thalach':'Max heart rate','exang':'Exercise-induced angina','oldpeak':'ST depression',
 'slope':'ST slope','ca':'Major vessels (fluoroscopy)','thal':'Thalassemia scan'}

report = {}
report['n_patients'] = int(len(df))
report['n_sites'] = len(SITES)
report['prevalence'] = round(df.target.mean()*100, 1)
report['site_counts'] = df.site.value_counts().to_dict()
report['site_prev'] = (df.groupby('site').target.mean()*100).round(1).to_dict()
report['missing'] = (df[COLS[:-1]].isna().mean()*100).round(1).sort_values(ascending=False).to_dict()

print(f"Loaded {len(df)} patients across {len(SITES)} sites")
print(f"Prevalence: {report['prevalence']}%")
print("\nMissingness (%):")
for k,v in report['missing'].items():
    if v>0: print(f"  {k:10s} {v:5.1f}")

# ============================================================
# 2. EDA FIGURES
# ============================================================

# --- Fig 1: cohort overview (site sizes + prevalence) ---
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
sc = df.site.value_counts()
axes[0].barh(sc.index[::-1], sc.values[::-1], color=C['secondary'], height=0.6)
for i,(n,v) in enumerate(zip(sc.index[::-1], sc.values[::-1])):
    axes[0].text(v+8, i, str(v), va='center', fontsize=9, color=C['dark'])
axes[0].set_title('Patients per study site', loc='left', fontweight='bold', pad=10)
axes[0].set_xlim(0, sc.max()*1.18); axes[0].grid(axis='y', alpha=0)

sp = df.groupby('site').target.mean().mul(100).sort_values()
cols = [C['ok'] if v<50 else C['primary'] for v in sp.values]
axes[1].barh(sp.index, sp.values, color=cols, height=0.6)
for i,v in enumerate(sp.values):
    axes[1].text(v+1.5, i, f'{v:.0f}%', va='center', fontsize=9, color=C['dark'])
axes[1].axvline(df.target.mean()*100, ls='--', lw=1.2, color=C['grey'])
axes[1].text(df.target.mean()*100+1, -0.65, 'pooled', fontsize=8, color=C['grey'])
axes[1].set_title('Disease prevalence by site', loc='left', fontweight='bold', pad=10)
axes[1].set_xlim(0, 105); axes[1].grid(axis='y', alpha=0)
plt.tight_layout(); plt.savefig(f'{OUT}/01_cohort.png', bbox_inches='tight'); plt.close()

# --- Fig 2: missingness ---
miss = df[COLS[:-1]].isna().mean().mul(100)
miss = miss[miss>0].sort_values()
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh([LABELS.get(i,i) for i in miss.index], miss.values,
        color=[C['primary'] if v>40 else C['accent'] if v>10 else C['grey'] for v in miss.values],
        height=0.6)
for i,v in enumerate(miss.values):
    ax.text(v+1, i, f'{v:.0f}%', va='center', fontsize=9)
ax.set_title('Missing data by variable', loc='left', fontweight='bold', pad=10)
ax.set_xlim(0, miss.max()*1.15); ax.grid(axis='y', alpha=0)
plt.tight_layout(); plt.savefig(f'{OUT}/02_missing.png', bbox_inches='tight'); plt.close()

# --- Fig 3: key continuous distributions by outcome ---
cont = ['age','thalach','oldpeak','trestbps']
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.2))
for ax, c in zip(axes, cont):
    for t, col, lab in [(0, C['ok'], 'No disease'), (1, C['primary'], 'Disease')]:
        s = df.loc[df.target==t, c].dropna()
        ax.hist(s, bins=22, alpha=0.55, color=col, label=lab, density=True)
    ax.set_title(LABELS[c], loc='left', fontsize=10, fontweight='bold')
    ax.set_yticks([])
axes[0].legend(frameon=False, fontsize=8.5)
plt.tight_layout(); plt.savefig(f'{OUT}/03_distributions.png', bbox_inches='tight'); plt.close()

# --- Fig 4: categorical risk rates ---
CAT_MAP = {
 'cp': {1:'Typical\nangina',2:'Atypical\nangina',3:'Non-anginal',4:'Asymptomatic'},
 'exang': {0:'No',1:'Yes'},
 'slope': {1:'Upsloping',2:'Flat',3:'Downsloping'},
 'sex': {0:'Female',1:'Male'}}
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
for ax, c in zip(axes, ['cp','exang','slope','sex']):
    g = df.groupby(c).agg(rate=('target','mean'), n=('target','size'))
    g = g[g.n>=15]
    names = [CAT_MAP[c].get(i, str(i)) for i in g.index]
    bars = ax.bar(names, g.rate*100,
                  color=[C['primary'] if v>0.55 else C['accent'] if v>0.35 else C['ok'] for v in g.rate],
                  width=0.62)
    for b,v,n in zip(bars, g.rate*100, g.n):
        ax.text(b.get_x()+b.get_width()/2, v+2, f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold')
        ax.text(b.get_x()+b.get_width()/2, 3, f'n={n}', ha='center', fontsize=7.5, color='white')
    ax.set_ylim(0,108); ax.set_title(LABELS[c], loc='left', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0); ax.tick_params(axis='x', labelsize=8.5)
axes[0].set_ylabel('% with disease')
plt.tight_layout(); plt.savefig(f'{OUT}/04_categorical.png', bbox_inches='tight'); plt.close()

# --- Fig 5: correlation ---
num = ['age','trestbps','chol','thalach','oldpeak','target']
corr = df[num].corr()
fig, ax = plt.subplots(figsize=(5.6, 4.6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-.6, vmax=.6, square=True, linewidths=1.2, linecolor='white',
            cbar_kws={'shrink':.7}, annot_kws={'size':9}, ax=ax)
ax.set_xticklabels([LABELS.get(i,'Disease') for i in num], rotation=35, ha='right', fontsize=8.5)
ax.set_yticklabels([LABELS.get(i,'Disease') for i in num], rotation=0, fontsize=8.5)
ax.set_title('Correlation matrix', loc='left', fontweight='bold', pad=12)
plt.tight_layout(); plt.savefig(f'{OUT}/05_corr.png', bbox_inches='tight'); plt.close()

# ============================================================
# 3. MODELLING  — site-aware split to test generalisation
# ============================================================
FEATS = COLS[:-1]
NUMF = ['age','trestbps','chol','thalach','oldpeak','ca']
CATF = ['sex','cp','fbs','restecg','exang','slope','thal']

pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler())]), NUMF),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('oh', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))]), CATF)])

MODELS = {
 'Logistic Regression': LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced'),
 'Decision Tree':       DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=42, class_weight='balanced'),
 'K-Nearest Neighbours':KNeighborsClassifier(n_neighbors=15, weights='distance'),
 'SVM (RBF)':           SVC(C=1.0, probability=True, random_state=42, class_weight='balanced'),
 'Random Forest':       RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5,
                                               random_state=42, class_weight='balanced', n_jobs=-1),
 'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                   learning_rate=0.06, random_state=42)}

X, y = df[FEATS], df.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

rows, fitted = [], {}
for name, clf in MODELS.items():
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    cvs = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring='roc_auc', n_jobs=-1)
    pipe.fit(Xtr, ytr)
    pr = pipe.predict(Xte); pp = pipe.predict_proba(Xte)[:,1]
    rows.append({'Model':name,'CV ROC-AUC':cvs.mean(),'CV std':cvs.std(),
                 'Test ROC-AUC':roc_auc_score(yte,pp),'Accuracy':accuracy_score(yte,pr),
                 'Precision':precision_score(yte,pr),'Recall':recall_score(yte,pr),
                 'F1':f1_score(yte,pr),'AP':average_precision_score(yte,pp)})
    fitted[name] = pipe
    print(f"{name:22s} CV {cvs.mean():.3f}  Test {roc_auc_score(yte,pp):.3f}  Recall {recall_score(yte,pr):.3f}")

res = pd.DataFrame(rows).sort_values('Test ROC-AUC', ascending=False).reset_index(drop=True)
best_name = res.iloc[0].Model; best = fitted[best_name]
report['results'] = res.round(3).to_dict('records')
report['best'] = best_name

# --- Fig 6: model comparison ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
o = res.sort_values('Test ROC-AUC')
ypos = np.arange(len(o))
axes[0].barh(ypos-0.19, o['CV ROC-AUC'], height=0.36, color=C['grey'], label='Cross-validation')
axes[0].barh(ypos+0.19, o['Test ROC-AUC'], height=0.36, color=C['secondary'], label='Held-out test')
axes[0].set_yticks(ypos); axes[0].set_yticklabels(o.Model, fontsize=9)
for i,v in enumerate(o['Test ROC-AUC']):
    axes[0].text(v+0.008, i+0.19, f'{v:.3f}', va='center', fontsize=8.5, fontweight='bold')
axes[0].set_xlim(0.5, 1.0); axes[0].legend(frameon=False, fontsize=8.5, loc='lower right')
axes[0].set_title('ROC-AUC by model', loc='left', fontweight='bold', pad=10)
axes[0].grid(axis='y', alpha=0)

for name in res.Model:
    pp = fitted[name].predict_proba(Xte)[:,1]
    fpr, tpr, _ = roc_curve(yte, pp)
    lw = 2.4 if name==best_name else 1.1
    col = C['primary'] if name==best_name else C['grey']
    a = 1.0 if name==best_name else 0.45
    axes[1].plot(fpr, tpr, lw=lw, color=col, alpha=a,
                 label=f'{name} ({roc_auc_score(yte,pp):.3f})' if name==best_name else None)
axes[1].plot([0,1],[0,1],'--',lw=1,color='#BBBBBB')
axes[1].set_xlabel('False positive rate'); axes[1].set_ylabel('True positive rate')
axes[1].legend(frameon=False, fontsize=9, loc='lower right')
axes[1].set_title('ROC curves', loc='left', fontweight='bold', pad=10)
plt.tight_layout(); plt.savefig(f'{OUT}/06_models.png', bbox_inches='tight'); plt.close()

# --- Fig 7: confusion matrix + threshold analysis ---
pp = best.predict_proba(Xte)[:,1]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))

cm = confusion_matrix(yte, (pp>=0.5).astype(int))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
            linewidths=2, linecolor='white', annot_kws={'size':15,'weight':'bold'}, ax=axes[0])
axes[0].set_xticklabels(['No disease','Disease']); axes[0].set_yticklabels(['No disease','Disease'], rotation=0)
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title(f'{best_name} — confusion matrix', loc='left', fontweight='bold', pad=10)

ths = np.linspace(0.05, 0.95, 91)
rec = [recall_score(yte,(pp>=t).astype(int), zero_division=0) for t in ths]
prc = [precision_score(yte,(pp>=t).astype(int), zero_division=0) for t in ths]
f1s = [f1_score(yte,(pp>=t).astype(int), zero_division=0) for t in ths]
axes[1].plot(ths, rec, color=C['primary'], lw=2, label='Recall (sensitivity)')
axes[1].plot(ths, prc, color=C['secondary'], lw=2, label='Precision')
axes[1].plot(ths, f1s, color=C['accent'], lw=1.6, ls='--', label='F1')
axes[1].axvline(0.5, color=C['grey'], ls=':', lw=1.2)
axes[1].text(0.51, 0.06, 'default 0.5', fontsize=8, color=C['grey'])
axes[1].set_xlabel('Decision threshold'); axes[1].set_ylim(0,1.02)
axes[1].legend(frameon=False, fontsize=8.5, loc='lower left')
axes[1].set_title('Threshold trade-off', loc='left', fontweight='bold', pad=10)

fr, mp = calibration_curve(yte, pp, n_bins=8, strategy='quantile')
axes[2].plot([0,1],[0,1],'--',lw=1,color='#BBBBBB')
axes[2].plot(mp, fr, 'o-', color=C['primary'], lw=2, ms=6)
axes[2].set_xlabel('Predicted probability'); axes[2].set_ylabel('Observed frequency')
axes[2].set_title('Calibration', loc='left', fontweight='bold', pad=10)
plt.tight_layout(); plt.savefig(f'{OUT}/07_diagnostics.png', bbox_inches='tight'); plt.close()

# threshold for 90% recall
idx90 = next((i for i,r in enumerate(rec) if r>=0.90), None)
if idx90 is not None:
    report['th90'] = {'threshold':round(float(ths[idx90]),2),
                      'recall':round(float(rec[idx90]),3),
                      'precision':round(float(prc[idx90]),3)}

# --- Fig 8: feature importance (permutation, model-agnostic) ---
pi = permutation_importance(best, Xte, yte, n_repeats=30, random_state=42,
                            scoring='roc_auc', n_jobs=-1)
imp = pd.DataFrame({'feat':FEATS,'mean':pi.importances_mean,'std':pi.importances_std}
                   ).sort_values('mean')
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.barh([LABELS[f] for f in imp.feat], imp['mean'], xerr=imp['std'],
        color=[C['primary'] if v>0.02 else C['secondary'] if v>0.005 else C['grey'] for v in imp['mean']],
        height=0.62, error_kw={'lw':0.8,'ecolor':'#999999'})
ax.axvline(0, color='#CCCCCC', lw=1)
ax.set_xlabel('Drop in ROC-AUC when feature is shuffled')
ax.set_title(f'What drives the prediction — {best_name}', loc='left', fontweight='bold', pad=10)
ax.grid(axis='y', alpha=0)
plt.tight_layout(); plt.savefig(f'{OUT}/08_importance.png', bbox_inches='tight'); plt.close()
report['importance'] = imp.sort_values('mean',ascending=False).head(6)[['feat','mean']].round(4).to_dict('records')

# ============================================================
# 4. GENERALISATION TEST — train on Cleveland, test elsewhere
# ============================================================
tr = df[df.site=='Cleveland']; gen_rows=[]
best_clf = MODELS[best_name]
g = Pipeline([('pre', pre), ('clf', best_clf)]).fit(tr[FEATS], tr.target)
for site in ['Hungary','Switzerland','Long Beach VA']:
    te = df[df.site==site]
    p = g.predict_proba(te[FEATS])[:,1]
    gen_rows.append({'Test site':site,'n':len(te),
                     'ROC-AUC':round(roc_auc_score(te.target,p),3),
                     'Recall':round(recall_score(te.target,(p>=0.5).astype(int)),3)})
report['generalisation'] = gen_rows
print("\nTrain on Cleveland, test elsewhere:")
for r in gen_rows: print(f"  {r['Test site']:15s} AUC {r['ROC-AUC']:.3f}  (n={r['n']})")

fig, ax = plt.subplots(figsize=(7, 3.2))
gs = pd.DataFrame(gen_rows)
b = ax.barh(gs['Test site'], gs['ROC-AUC'], color=C['secondary'], height=0.55)
ax.axvline(res.iloc[0]['Test ROC-AUC'], ls='--', color=C['primary'], lw=1.5)
ax.text(res.iloc[0]['Test ROC-AUC']-0.01, 2.55, 'pooled random split', fontsize=8,
        color=C['primary'], ha='right')
for i,v in enumerate(gs['ROC-AUC']): ax.text(v+0.008, i, f'{v:.3f}', va='center', fontsize=9)
ax.set_xlim(0.5, 1.0); ax.grid(axis='y', alpha=0)
ax.set_title('Trained on Cleveland only — tested on unseen sites', loc='left', fontweight='bold', pad=10)
plt.tight_layout(); plt.savefig(f'{OUT}/09_generalisation.png', bbox_inches='tight'); plt.close()

# ============================================================
# 5. SYNTHETIC-DATA CHECK
# ============================================================
syn = pd.read_csv('/home/claude/work/kaggle_ds/heart_disease_dataset.csv')
from sklearn.preprocessing import LabelEncoder
Xs = syn.drop(columns=['patient_id','heart_disease','treatment','source']).copy()
for c in Xs.columns:
    if Xs[c].dtype == object or str(Xs[c].dtype) == 'str':
        Xs[c] = LabelEncoder().fit_transform(Xs[c].astype(str))
syn_auc = cross_val_score(RandomForestClassifier(n_estimators=400, random_state=42),
                          Xs, syn.heart_disease, cv=cv, scoring='roc_auc')
report['synthetic_auc'] = round(float(syn_auc.mean()), 3)
report['real_auc'] = round(float(res.iloc[0]['Test ROC-AUC']), 3)

fig, ax = plt.subplots(figsize=(6.4, 3))
vals = [syn_auc.mean(), res.iloc[0]['Test ROC-AUC']]
bars = ax.barh(['Synthetic Kaggle file\n(500 rows)','Real UCI data\n(920 patients)'], vals,
               color=[C['grey'], C['ok']], height=0.5)
ax.axvline(0.5, ls='--', color=C['primary'], lw=1.4)
ax.text(0.505, 1.45, 'random guessing', fontsize=8, color=C['primary'])
for i,v in enumerate(vals): ax.text(v+0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlim(0.4, 1.0); ax.set_xlabel('ROC-AUC'); ax.grid(axis='y', alpha=0)
ax.set_title('Why the dataset choice matters', loc='left', fontweight='bold', pad=10)
plt.tight_layout(); plt.savefig(f'{OUT}/10_synthetic.png', bbox_inches='tight'); plt.close()

# ---- save ----
df.to_csv('/home/claude/work/heart_disease_merged.csv', index=False)
res.round(4).to_csv('/home/claude/work/model_results.csv', index=False)
with open('/home/claude/work/report.json','w') as f: json.dump(report, f, indent=2, default=str)

print(f"\nBest model: {best_name}")
print(f"Synthetic AUC {report['synthetic_auc']} vs Real AUC {report['real_auc']}")
print("Done.")
