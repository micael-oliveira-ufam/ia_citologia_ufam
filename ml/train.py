"""Treino e comparação de algoritmos para estimar risco de câncer de colo do útero.

Dataset: UCI "Cervical Cancer (Risk Factors)" – Hospital Universitário de Caracas
(858 pacientes, 36 variáveis). Alvo: Biopsy (padrão-ouro). Usamos apenas variáveis
que podem ser respondidas em um questionário (perfil, hábitos, história sexual e
reprodutiva, contracepção, IST). Resultados de exames (Hinselmann, Schiller,
Citologia) e diagnósticos (Dx:*) ficam de FORA para evitar vazamento de alvo.

Uso:  python3 train.py   (gera ml/outputs/*, app/assets/risco/modelo_risco.json,
      app/assets/graficos/*.png e app/test/casos_referencia.csv)
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
APP = ROOT.parent / "app"
RISCO = APP / "assets" / "risco"
GRAFICOS = APP / "assets" / "graficos"
OUT.mkdir(exist_ok=True)
RISCO.mkdir(parents=True, exist_ok=True)
GRAFICOS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- dados
df = pd.read_csv(ROOT / "data_raw" / "risk_factors_cervical_cancer.csv", na_values="?")

FEATURES = {  # coluna original -> nome curto usado no app
    "Age": "age",
    "Number of sexual partners": "partners",
    "First sexual intercourse": "first_intercourse",
    "Num of pregnancies": "pregnancies",
    "Smokes": "smokes",
    "Smokes (years)": "smokes_years",
    "Hormonal Contraceptives": "hormonal",
    "Hormonal Contraceptives (years)": "hormonal_years",
    "IUD": "iud",
    "IUD (years)": "iud_years",
    "STDs": "stds",
    "STDs (number)": "stds_number",
}
TARGET = "Biopsy"
X = df[list(FEATURES)].rename(columns=FEATURES)
y = df[TARGET].values
feat_names = list(X.columns)
print(f"n={len(X)}  positivos={y.sum()} ({y.mean():.1%})  features={len(feat_names)}")

# ---------------------------------------------------------------- modelos
def pipe(clf):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", clf),
    ])

class BalancedGB(GradientBoostingClassifier):
    """Gradient Boosting com pesos de amostra inversamente proporcionais à frequência da classe."""
    def fit(self, X, y, sample_weight=None):
        from sklearn.utils.class_weight import compute_sample_weight
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y))


MODELS = {
    "Regressão Logística": LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=SEED),
    "Árvore de Decisão": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                            class_weight="balanced", n_jobs=-1, random_state=SEED),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                        class_weight="balanced", n_jobs=-1, random_state=SEED),
    "Gradient Boosting": BalancedGB(n_estimators=100, max_depth=2,
                                         learning_rate=0.05, random_state=SEED),
    "SVM (RBF)": SVC(C=1.0, probability=True, class_weight="balanced", random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "MLP (rede neural)": MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1e-2, max_iter=1500,
                                       random_state=SEED),
    "Naive Bayes": GaussianNB(),
}
EXPORTABLE = {"Regressão Logística", "Random Forest", "Extra Trees", "Gradient Boosting"}

# ---------------------------------------------------------------- validação cruzada
N_SPLITS, N_REPEATS = 5, 5
cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
splits = list(cv.split(X, y))

def fold_metrics(yt, yp, pr):
    return dict(
        accuracy=accuracy_score(yt, yp),
        balanced_accuracy=balanced_accuracy_score(yt, yp),
        recall=recall_score(yt, yp, zero_division=0),
        precision=precision_score(yt, yp, zero_division=0),
        f1=f1_score(yt, yp, zero_division=0),
        roc_auc=roc_auc_score(yt, pr),
        pr_auc=average_precision_score(yt, pr),
    )

rows, oof = [], {}
all_models = {"Baseline (sempre 'sem risco')": DummyClassifier(strategy="most_frequent")}
all_models.update({k: pipe(v) for k, v in MODELS.items()})
for name, model in all_models.items():
    from sklearn.base import clone
    oof_sum = np.zeros(len(y)); oof_cnt = np.zeros(len(y))
    for tr, te in splits:
        m = clone(model).fit(X.iloc[tr], y[tr])
        pr = m.predict_proba(X.iloc[te])[:, 1]
        yp = m.predict(X.iloc[te])
        fm = fold_metrics(y[te], yp, pr if name.startswith("Baseline") is False else np.zeros(len(te)) + 0.5 * 0)
        if name.startswith("Baseline"):
            fm["roc_auc"], fm["pr_auc"] = 0.5, y.mean()
        rows.append(dict(model=name, **fm))
        oof_sum[te] += pr; oof_cnt[te] += 1
    oof[name] = oof_sum / oof_cnt
    print(f"{name:32s} ok")

res = pd.DataFrame(rows)
summary = res.groupby("model").agg(["mean", "std"])
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary["oof_auc"] = pd.Series({n: (0.5 if n.startswith("Baseline") else roc_auc_score(y, oof[n])) for n in oof})
summary = summary.sort_values("oof_auc", ascending=False)
summary.round(4).to_csv(OUT / "metricas_cv.csv")
print("\n", summary[["accuracy_mean", "balanced_accuracy_mean", "recall_mean", "oof_auc"]].round(3).to_string())

# modelo final (entre os exportáveis): maior AUC médio; desempate por recall
cand = summary.loc[[m for m in summary.index if m in EXPORTABLE]]
best_name = cand.sort_values(["oof_auc", "balanced_accuracy_mean"], ascending=False).index[0]
print("\nModelo exportado para o app:", best_name)

# ---------------------------------------------------------------- gráficos
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a8a7a2"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})

def clean(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)

order = summary.index.tolist()

# 1) acurácia por modelo (com baseline)
s = summary.sort_values("accuracy_mean")
fig, ax = plt.subplots(figsize=(8.5, 5.2)); clean(ax)
cols = [GRAY if n.startswith("Baseline") else BLUE for n in s.index]
ax.barh(s.index, s["accuracy_mean"] * 100, xerr=s["accuracy_std"] * 100, color=cols, height=0.6,
        error_kw=dict(ecolor=INK2, lw=1, capsize=3))
for i, (v, n) in enumerate(zip(s["accuracy_mean"] * 100, s.index)):
    ax.text(v + s["accuracy_std"].iloc[i] * 100 + 0.8, i, f"{v:.1f}%", va="center", fontsize=9, color=INK)
ax.set_xlim(0, 108); ax.set_xlabel("Acurácia (%) – média ± desvio em 5×5 validação cruzada")
ax.set_title("Acurácia por algoritmo", loc="left", fontsize=13, fontweight="bold")
fig.text(0.01, 0.01, "Cinza = baseline que nunca prevê risco: já acerta ~94% porque só 6,4% são positivas. "
         "Acurácia sozinha engana.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "01_acuracia.png", dpi=170); plt.close(fig)

# 2) acurácia x acurácia balanceada x recall
s = summary.drop(index=[i for i in summary.index if i.startswith("Baseline")]).sort_values("balanced_accuracy_mean")
fig, ax = plt.subplots(figsize=(9, 5.6)); clean(ax)
h = 0.26; y0 = np.arange(len(s))
for k, (col, lab, c) in enumerate([("accuracy_mean", "Acurácia", BLUE),
                                   ("balanced_accuracy_mean", "Acurácia balanceada", ORANGE),
                                   ("recall_mean", "Sensibilidade (recall)", AQUA)]):
    ax.barh(y0 + (k - 1) * h, s[col] * 100, height=h - 0.03, color=c, label=lab)
    for yy, v in zip(y0 + (k - 1) * h, s[col] * 100):
        ax.text(v + 0.8, yy, f"{v:.0f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(y0); ax.set_yticklabels(s.index); ax.set_xlim(0, 105)
ax.set_xlabel("%"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=9)
ax.set_title("Acurácia vs. acurácia balanceada vs. sensibilidade", loc="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "02_acuracia_balanceada_recall.png", dpi=170); plt.close(fig)

# 3) ROC-AUC por modelo
s = summary.drop(index=[i for i in summary.index if i.startswith("Baseline")]).sort_values("oof_auc")
fig, ax = plt.subplots(figsize=(8.5, 5)); clean(ax)
ax.barh(s.index, s["oof_auc"], color=BLUE, height=0.6)
ax.axvline(0.5, color=GRAY, lw=1, ls="--"); ax.text(0.505, -0.75, "acaso = 0,5", fontsize=8, color=INK2)
for i, v in enumerate(s["oof_auc"]):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0.4, 1.0); ax.set_xlabel("ROC-AUC das predições fora da amostra (validação cruzada 5×5)")
ax.set_title("Capacidade de discriminar casos (ROC-AUC)", loc="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "03_roc_auc.png", dpi=170); plt.close(fig)

# 4) curvas ROC (predições out-of-fold, 4 melhores destacados)
top4 = [m for m in order if not m.startswith("Baseline")][:4]
palette = [BLUE, ORANGE, AQUA, "#eda100"]
fig, ax = plt.subplots(figsize=(6.4, 6)); ax.set_facecolor(SURF)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for n in oof:
    if n.startswith("Baseline") or n in top4: continue
    f, t, _ = roc_curve(y, oof[n]); ax.plot(f, t, color=GRAY, lw=1, alpha=0.6)
for n, c in zip(top4, palette):
    f, t, _ = roc_curve(y, oof[n])
    ax.plot(f, t, color=c, lw=2, label=f"{n} (AUC {roc_auc_score(y, oof[n]):.2f})")
ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
ax.set_xlabel("1 − especificidade"); ax.set_ylabel("Sensibilidade")
ax.legend(loc="lower right", frameon=False, fontsize=8.5)
ax.set_title("Curvas ROC (predições fora da amostra)", loc="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "04_curvas_roc.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- tiers de risco (a partir das predições OOF)
p = oof[best_name]
q_low, q_high = np.quantile(p, [0.60, 0.85])
tier = np.where(p >= q_high, 2, np.where(p >= q_low, 1, 0))
tiers = []
for k, lab in enumerate(["baixo", "moderado", "alto"]):
    m = tier == k
    tiers.append(dict(level=lab, n=int(m.sum()), positives=int(y[m].sum()),
                      rate=float(y[m].mean())))
print("\nTiers (OOF):", json.dumps(tiers, ensure_ascii=False))

# 5) matriz de confusão do modelo final (limiar = corte "alto" ∪ "moderado")
yp = (p >= q_low).astype(int)
cm = confusion_matrix(y, yp)
fig, ax = plt.subplots(figsize=(4.8, 4.2)); ax.set_facecolor(SURF)
ax.imshow(cm, cmap="Blues"); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["baixo risco", "risco ≥ moderado"]); ax.set_yticklabels(["biópsia −", "biópsia +"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else INK)
ax.set_title(f"{best_name}\n(predição fora da amostra)", loc="left", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "05_matriz_confusao.png", dpi=170); plt.close(fig)

# teste de permutação: AUC com rótulos embaralhados (mesmo modelo, mesmo CV)
from sklearn.base import clone as _clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
rng = np.random.default_rng(0)
null = []
for _ in range(30):
    ys = rng.permutation(y)
    pp = cross_val_predict(_clone(pipe(MODELS[best_name])), X, ys,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    null.append(roc_auc_score(ys, pp))
perm = dict(auc_real=float(roc_auc_score(y, oof[best_name])), null_mean=float(np.mean(null)),
            null_std=float(np.std(null)), p_value=float((np.sum(np.array(null) >= roc_auc_score(y, oof[best_name])) + 1) / (len(null) + 1)))
print("Permutação:", perm)

# ---------------------------------------------------------------- modelo final + exportação
final = pipe(MODELS[best_name]).fit(X, y)
imp, sc, clf = final.named_steps["imp"], final.named_steps["sc"], final.named_steps["clf"]

# 6) importância (permutação no OOF não é trivial; usa a do modelo quando existe, senão |coef|)
if hasattr(clf, "feature_importances_"):
    imps = clf.feature_importances_
elif hasattr(clf, "coef_"):
    imps = np.abs(clf.coef_[0]) / np.abs(clf.coef_[0]).sum()
order_i = np.argsort(imps)
fig, ax = plt.subplots(figsize=(7.5, 4.6)); clean(ax)
ax.barh(np.array(feat_names)[order_i], imps[order_i], color=BLUE, height=0.6)
ax.set_xlabel("Importância relativa"); ax.set_title(f"Variáveis mais influentes – {best_name}",
                                                    loc="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "06_importancia.png", dpi=170); plt.close(fig)

def tree_to_dict(t, value_fn):
    return dict(left=t.children_left.tolist(), right=t.children_right.tolist(),
                feature=t.feature.tolist(), threshold=[float(x) for x in t.threshold],
                value=[float(value_fn(i)) for i in range(t.node_count)])

model_json = dict(
    name=best_name, features=feat_names,
    impute_median=[float(x) for x in imp.statistics_],
    scale_mean=[float(x) for x in sc.mean_], scale_std=[float(x) for x in sc.scale_],
    tiers=dict(cut_moderate=float(q_low), cut_high=float(q_high), observed=tiers),
    validation=dict(cv="5x5 estratificada", n=int(len(y)), positives=int(y.sum()),
                    oof_auc=perm["auc_real"], permutation=perm),
)
if best_name == "Regressão Logística":
    model_json.update(type="logistic", coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]))
elif best_name in ("Random Forest", "Extra Trees"):
    trees = []
    for est in clf.estimators_:
        t = est.tree_
        trees.append(tree_to_dict(t, lambda i: t.value[i][0][1] / t.value[i][0].sum()))
    model_json.update(type="forest", trees=trees)
elif best_name == "Gradient Boosting":
    init_p = clf.init_.class_prior_[1]
    trees = [tree_to_dict(e[0].tree_, lambda i, e=e: e[0].tree_.value[i][0][0]) for e in clf.estimators_]
    model_json.update(type="gbm", init=float(np.log(init_p / (1 - init_p))),
                      learning_rate=float(clf.learning_rate), trees=trees)
(RISCO / "modelo_risco.json").write_text(json.dumps(model_json, separators=(",", ":")))
(OUT / "model_meta.json").write_text(json.dumps(
    {k: v for k, v in model_json.items() if k != "trees"}, ensure_ascii=False, indent=1))

# referência para testar a implementação em Dart: 30 casos com a prob. exata do modelo
rng = np.random.default_rng(1)
ref = X.sample(30, random_state=1).copy()
ref["prob"] = final.predict_proba(ref[feat_names])[:, 1]
ref.to_csv(OUT / "casos_referencia.csv", index=False)
(APP / "test").mkdir(exist_ok=True)
ref.to_csv(APP / "test" / "casos_referencia.csv", index=False)

# gráficos exibidos no app (aba "Modelo de risco", perfil profissional)
import shutil
for nome in ("01_acuracia", "02_acuracia_balanceada_recall", "03_roc_auc", "04_curvas_roc", "06_importancia"):
    shutil.copy(OUT / f"{nome}.png", GRAFICOS / f"{nome}.png")
print("modelo_risco.json:", (RISCO / "modelo_risco.json").stat().st_size // 1024, "KB")
print("OK")
