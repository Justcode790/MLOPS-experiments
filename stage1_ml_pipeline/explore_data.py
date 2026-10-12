"""Quick data exploration (shows the TA what the dataset looks like)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import HERE, load_data

df = load_data()
print("Shape:", df.shape)
print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])
print("\nSurvival rate overall: %.1f%%" % (df.Survived.mean() * 100))
print("\nSurvival rate by Sex:\n", df.groupby("Sex").Survived.mean().round(3))
print("\nSurvival rate by Pclass:\n", df.groupby("Pclass").Survived.mean().round(3))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
df.groupby("Sex").Survived.mean().plot.bar(ax=ax[0], title="Survival rate by sex", color="#4C78A8")
df.groupby("Pclass").Survived.mean().plot.bar(ax=ax[1], title="Survival rate by class", color="#F58518")
for a in ax:
    a.set_ylabel("survival rate"); a.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(HERE / "eda_survival.png", dpi=120)
print("\nSaved eda_survival.png")
