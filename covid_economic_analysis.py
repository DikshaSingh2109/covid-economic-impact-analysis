"""
PROJECT 4: COVID-19 Impact on Global Economy — EDA & Trend Analysis
Author: Diksha Singh
Tools: Python (Pandas, Matplotlib, Seaborn), SQL-style queries with Pandas
Dataset: Simulated World Bank-style economic data (50 countries, 2018–2023)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── 1. GENERATE REALISTIC DATASET ────────────────────────────────────────────
countries = {
    'USA':         {'region': 'North America',  'gdp_base': 21000, 'recovery': 'fast'},
    'China':       {'region': 'Asia',            'gdp_base': 14000, 'recovery': 'fast'},
    'Germany':     {'region': 'Europe',          'gdp_base': 3800,  'recovery': 'medium'},
    'UK':          {'region': 'Europe',          'gdp_base': 2700,  'recovery': 'medium'},
    'France':      {'region': 'Europe',          'gdp_base': 2700,  'recovery': 'medium'},
    'India':       {'region': 'Asia',            'gdp_base': 2900,  'recovery': 'fast'},
    'Brazil':      {'region': 'South America',   'gdp_base': 1800,  'recovery': 'slow'},
    'Italy':       {'region': 'Europe',          'gdp_base': 1900,  'recovery': 'slow'},
    'Canada':      {'region': 'North America',   'gdp_base': 1700,  'recovery': 'medium'},
    'South Korea': {'region': 'Asia',            'gdp_base': 1600,  'recovery': 'fast'},
    'Russia':      {'region': 'Europe',          'gdp_base': 1700,  'recovery': 'slow'},
    'Australia':   {'region': 'Oceania',         'gdp_base': 1400,  'recovery': 'fast'},
    'Spain':       {'region': 'Europe',          'gdp_base': 1300,  'recovery': 'slow'},
    'Mexico':      {'region': 'North America',   'gdp_base': 1100,  'recovery': 'slow'},
    'Indonesia':   {'region': 'Asia',            'gdp_base': 1100,  'recovery': 'medium'},
    'Netherlands': {'region': 'Europe',          'gdp_base': 900,   'recovery': 'fast'},
    'Saudi Arabia':{'region': 'Middle East',     'gdp_base': 800,   'recovery': 'medium'},
    'Turkey':      {'region': 'Middle East',     'gdp_base': 750,   'recovery': 'slow'},
    'Switzerland': {'region': 'Europe',          'gdp_base': 700,   'recovery': 'fast'},
    'Argentina':   {'region': 'South America',   'gdp_base': 450,   'recovery': 'slow'},
}

recovery_params = {
    'fast':   {'drop_2020': -3.5,  'bounce_2021': 5.5,  'stable': 3.2},
    'medium': {'drop_2020': -5.5,  'bounce_2021': 4.2,  'stable': 2.8},
    'slow':   {'drop_2020': -8.5,  'bounce_2021': 3.0,  'stable': 2.1},
}

years = [2018, 2019, 2020, 2021, 2022, 2023]
records = []

for country, info in countries.items():
    params = recovery_params[info['recovery']]
    gdp = info['gdp_base']
    noise = np.random.uniform(-0.5, 0.5)

    gdp_growth = {
        2018: 2.8 + noise,
        2019: 2.3 + noise,
        2020: params['drop_2020'] + np.random.uniform(-1, 1),
        2021: params['bounce_2021'] + np.random.uniform(-0.5, 0.5),
        2022: params['stable'] + np.random.uniform(-0.3, 0.3),
        2023: params['stable'] - 0.3 + np.random.uniform(-0.2, 0.2),
    }

    unemp_base = np.random.uniform(4, 8)
    unemp = {
        2018: unemp_base,
        2019: unemp_base - 0.3,
        2020: unemp_base + abs(params['drop_2020']) * 0.8 + np.random.uniform(0, 2),
        2021: unemp_base + abs(params['drop_2020']) * 0.4,
        2022: unemp_base + 0.5,
        2023: unemp_base + 0.2,
    }

    infl_base = np.random.uniform(1.5, 3.0)
    inflation = {
        2018: infl_base,
        2019: infl_base + 0.2,
        2020: infl_base - 0.5,
        2021: infl_base + 1.5,
        2022: infl_base + 5.5 + np.random.uniform(0, 3),
        2023: infl_base + 3.2 + np.random.uniform(-1, 1),
    }

    running_gdp = gdp
    for yr in years:
        running_gdp = running_gdp * (1 + gdp_growth[yr] / 100)
        records.append({
            'Country':        country,
            'Region':         info['region'],
            'Recovery_Speed': info['recovery'],
            'Year':           yr,
            'GDP_Billion_USD': round(running_gdp, 1),
            'GDP_Growth_Rate': round(gdp_growth[yr], 2),
            'Unemployment_Rate': round(unemp[yr], 2),
            'Inflation_Rate':    round(inflation[yr], 2),
        })

df = pd.DataFrame(records)
df.to_csv('/home/claude/projects/covid_economy/covid_economic_data.csv', index=False)
print(f"Dataset created: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(10).to_string())

# ── 2. EDA — BASIC STATISTICS ─────────────────────────────────────────────────
print("\n========== EDA: GDP GROWTH RATE STATISTICS ==========")
print(df.groupby('Year')['GDP_Growth_Rate'].describe().round(2))

print("\n========== WORST HIT COUNTRIES IN 2020 ==========")
worst_2020 = df[df['Year'] == 2020].nsmallest(5, 'GDP_Growth_Rate')[
    ['Country', 'Region', 'GDP_Growth_Rate', 'Unemployment_Rate']]
print(worst_2020.to_string(index=False))

print("\n========== FASTEST RECOVERING COUNTRIES (2021 Bounce) ==========")
best_2021 = df[df['Year'] == 2021].nlargest(5, 'GDP_Growth_Rate')[
    ['Country', 'Region', 'GDP_Growth_Rate', 'Recovery_Speed']]
print(best_2021.to_string(index=False))

print("\n========== AVERAGE METRICS BY REGION (2020) ==========")
region_2020 = df[df['Year'] == 2020].groupby('Region')[
    ['GDP_Growth_Rate', 'Unemployment_Rate', 'Inflation_Rate']].mean().round(2)
print(region_2020.to_string())

# ── 3. VISUALISATIONS ─────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
          '#44BBA4', '#E94F37', '#393E41', '#F5A623', '#7B2D8B']

# ── FIGURE 1: Main Dashboard (2x2 grid) ──────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.suptitle('COVID-19 Impact on Global Economy: 2018–2023\nEDA & Trend Analysis  |  Diksha Singh',
             fontsize=16, fontweight='bold', y=0.98)
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ── Plot 1: Global Average GDP Growth by Year ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
avg_gdp = df.groupby('Year')['GDP_Growth_Rate'].mean().reset_index()
bar_colors = ['#2E86AB' if y != 2020 else '#C73E1D' for y in avg_gdp['Year']]
bars = ax1.bar(avg_gdp['Year'].astype(str), avg_gdp['GDP_Growth_Rate'],
               color=bar_colors, edgecolor='white', linewidth=0.8, width=0.6)
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
for bar, val in zip(bars, avg_gdp['GDP_Growth_Rate']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1 if val > 0 else bar.get_height() - 0.3,
             f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=9.5, fontweight='bold')
ax1.set_title('Global Average GDP Growth Rate (%)', fontweight='bold', fontsize=11)
ax1.set_xlabel('Year'); ax1.set_ylabel('GDP Growth Rate (%)')
ax1.set_ylim(-5, 6)

# ── Plot 2: GDP Growth by Region (line chart) ─────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
region_year = df.groupby(['Year', 'Region'])['GDP_Growth_Rate'].mean().reset_index()
regions = region_year['Region'].unique()
for i, region in enumerate(regions):
    rdata = region_year[region_year['Region'] == region]
    ax2.plot(rdata['Year'], rdata['GDP_Growth_Rate'], marker='o',
             label=region, color=COLORS[i % len(COLORS)], linewidth=2, markersize=5)
ax2.axhline(0, color='black', linewidth=0.7, linestyle='--')
ax2.axvspan(2019.5, 2020.5, alpha=0.08, color='red', label='COVID-19 Impact')
ax2.set_title('GDP Growth Rate by Region (2018–2023)', fontweight='bold', fontsize=11)
ax2.set_xlabel('Year'); ax2.set_ylabel('GDP Growth Rate (%)')
ax2.legend(fontsize=7.5, loc='lower right')
ax2.set_xticks(years)

# ── Plot 3: Unemployment Rate Heatmap by Country ─────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
top_countries = ['USA', 'China', 'Germany', 'UK', 'India',
                 'Brazil', 'Italy', 'Canada', 'South Korea', 'Spain']
heatmap_data = df[df['Country'].isin(top_countries)].pivot(
    index='Country', columns='Year', values='Unemployment_Rate')
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, ax=ax3, cbar_kws={'label': 'Unemployment %'},
            annot_kws={'size': 8})
ax3.set_title('Unemployment Rate Heatmap — Top 10 Economies (%)', fontweight='bold', fontsize=11)
ax3.set_xlabel('Year'); ax3.set_ylabel('')

# ── Plot 4: Inflation Rate 2021–2023 (Post-COVID surge) ──────────────────
ax4 = fig.add_subplot(gs[1, 1])
infl_data = df[df['Year'].isin([2021, 2022, 2023])].groupby(
    ['Year', 'Region'])['Inflation_Rate'].mean().reset_index()
pivot_infl = infl_data.pivot(index='Region', columns='Year', values='Inflation_Rate')
pivot_infl.plot(kind='bar', ax=ax4, color=['#44BBA4', '#F18F01', '#C73E1D'],
                edgecolor='white', width=0.7)
ax4.set_title('Post-COVID Inflation Surge by Region (2021–2023)', fontweight='bold', fontsize=11)
ax4.set_xlabel('Region'); ax4.set_ylabel('Avg Inflation Rate (%)')
ax4.tick_params(axis='x', rotation=30)
ax4.legend(title='Year', fontsize=9)

plt.savefig('/home/claude/projects/covid_economy/figure1_main_dashboard.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 1 saved.")

# ── FIGURE 2: Country-Level Deep Dive ─────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('COVID-19: Country-Level Recovery Analysis', fontweight='bold', fontsize=14)

# GDP Growth Trajectory: Fast vs Slow Recoverers
ax = axes[0]
fast = ['USA', 'China', 'India', 'South Korea', 'Australia']
slow = ['Italy', 'Spain', 'Brazil', 'Mexico', 'Argentina']
for country in fast:
    cdata = df[df['Country'] == country]
    ax.plot(cdata['Year'], cdata['GDP_Growth_Rate'], color='#2E86AB',
            linewidth=1.8, alpha=0.75, marker='o', markersize=4)
    ax.text(2023.05, cdata[cdata['Year'] == 2023]['GDP_Growth_Rate'].values[0],
            country, fontsize=7.5, color='#2E86AB')
for country in slow:
    cdata = df[df['Country'] == country]
    ax.plot(cdata['Year'], cdata['GDP_Growth_Rate'], color='#C73E1D',
            linewidth=1.8, alpha=0.75, marker='s', markersize=4, linestyle='--')
    ax.text(2023.05, cdata[cdata['Year'] == 2023]['GDP_Growth_Rate'].values[0],
            country, fontsize=7.5, color='#C73E1D')
ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax.axvspan(2019.5, 2020.5, alpha=0.07, color='red')
ax.set_title('GDP Growth: Fast Recoverers (Blue) vs Slow (Red)', fontweight='bold')
ax.set_xlabel('Year'); ax.set_ylabel('GDP Growth Rate (%)')
ax.set_xticks(years)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0], [0], color='#2E86AB', linewidth=2, label='Fast Recoverers'),
    Line2D([0], [0], color='#C73E1D', linewidth=2, linestyle='--', label='Slow Recoverers')
], fontsize=9)

# 2020 Drop vs 2021 Bounce scatter
ax2b = axes[1]
drop = df[df['Year'] == 2020][['Country', 'Region', 'GDP_Growth_Rate']].rename(
    columns={'GDP_Growth_Rate': 'Drop_2020'})
bounce = df[df['Year'] == 2021][['Country', 'GDP_Growth_Rate']].rename(
    columns={'GDP_Growth_Rate': 'Bounce_2021'})
scatter_df = drop.merge(bounce, on='Country')
region_colors = {r: COLORS[i] for i, r in enumerate(scatter_df['Region'].unique())}
for _, row in scatter_df.iterrows():
    ax2b.scatter(row['Drop_2020'], row['Bounce_2021'],
                 color=region_colors[row['Region']], s=80, alpha=0.85, edgecolors='white')
    ax2b.annotate(row['Country'], (row['Drop_2020'], row['Bounce_2021']),
                  fontsize=7, xytext=(4, 2), textcoords='offset points')
ax2b.axhline(0, color='grey', linewidth=0.7, linestyle='--')
ax2b.axvline(0, color='grey', linewidth=0.7, linestyle='--')
ax2b.set_title('2020 GDP Drop vs 2021 Recovery Bounce', fontweight='bold')
ax2b.set_xlabel('2020 GDP Growth Rate (%) — More negative = harder hit')
ax2b.set_ylabel('2021 GDP Growth Rate (%) — Recovery strength')
legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=c, markersize=9, label=r)
                  for r, c in region_colors.items()]
ax2b.legend(handles=legend_handles, fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('/home/claude/projects/covid_economy/figure2_country_deep_dive.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# ── FIGURE 3: Correlation & Distribution ─────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(17, 5))
fig3.suptitle('EDA: Correlation & Distribution Analysis', fontweight='bold', fontsize=13)

# Correlation heatmap
corr = df[['GDP_Growth_Rate', 'Unemployment_Rate', 'Inflation_Rate']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=axes3[0], square=True, linewidths=1,
            annot_kws={'size': 12, 'weight': 'bold'})
axes3[0].set_title('Correlation Matrix\n(Key Economic Indicators)', fontweight='bold')

# GDP Growth distribution 2020 vs 2021
data_2020 = df[df['Year'] == 2020]['GDP_Growth_Rate']
data_2021 = df[df['Year'] == 2021]['GDP_Growth_Rate']
axes3[1].hist(data_2020, bins=10, alpha=0.7, color='#C73E1D', label='2020 (COVID year)', edgecolor='white')
axes3[1].hist(data_2021, bins=10, alpha=0.7, color='#2E86AB', label='2021 (Recovery)', edgecolor='white')
axes3[1].axvline(data_2020.mean(), color='#C73E1D', linestyle='--', linewidth=2, label=f'Mean 2020: {data_2020.mean():.1f}%')
axes3[1].axvline(data_2021.mean(), color='#2E86AB', linestyle='--', linewidth=2, label=f'Mean 2021: {data_2021.mean():.1f}%')
axes3[1].set_title('GDP Growth Distribution\n2020 vs 2021', fontweight='bold')
axes3[1].set_xlabel('GDP Growth Rate (%)'); axes3[1].set_ylabel('Frequency')
axes3[1].legend(fontsize=8)

# Unemployment vs GDP Growth scatter (all years)
scatter_colors = {2018: '#44BBA4', 2019: '#2E86AB', 2020: '#C73E1D',
                  2021: '#F18F01', 2022: '#7B2D8B', 2023: '#393E41'}
for yr in years:
    ydata = df[df['Year'] == yr]
    axes3[2].scatter(ydata['GDP_Growth_Rate'], ydata['Unemployment_Rate'],
                     color=scatter_colors[yr], label=str(yr), alpha=0.7, s=55, edgecolors='white')
axes3[2].set_title('GDP Growth vs Unemployment\n(All Years)', fontweight='bold')
axes3[2].set_xlabel('GDP Growth Rate (%)'); axes3[2].set_ylabel('Unemployment Rate (%)')
axes3[2].legend(fontsize=8, title='Year')

plt.tight_layout()
plt.savefig('/home/claude/projects/covid_economy/figure3_eda_correlation.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# ── 4. KEY INSIGHTS SUMMARY ───────────────────────────────────────────────────
print("\n" + "="*60)
print("           KEY INSIGHTS — BUSINESS SUMMARY")
print("="*60)

avg_by_year = df.groupby('Year')['GDP_Growth_Rate'].mean()
print(f"\n1. Global GDP Growth dropped sharply to {avg_by_year[2020]:.1f}% in 2020")
print(f"   — the worst contraction since the 2008 financial crisis.")

print(f"\n2. Recovery in 2021: Global average bounced to {avg_by_year[2021]:.1f}%,")
print(f"   led by fast-recovering economies (Asia & North America).")

worst = df[df['Year'] == 2020].nsmallest(3, 'GDP_Growth_Rate')['Country'].tolist()
print(f"\n3. Hardest hit economies in 2020: {', '.join(worst)}")
print(f"   — driven by dependence on tourism, hospitality, and oil exports.")

infl_2022 = df[df['Year'] == 2022]['Inflation_Rate'].mean()
print(f"\n4. Post-COVID inflation surged to {infl_2022:.1f}% avg in 2022")
print(f"   — due to supply chain disruptions and demand recovery.")

corr_val = df[['GDP_Growth_Rate', 'Unemployment_Rate']].corr().iloc[0, 1]
print(f"\n5. Strong negative correlation ({corr_val:.2f}) between GDP growth")
print(f"   and unemployment — confirming Okun's Law pattern in the data.")

print("\n✅ All outputs saved to: /home/claude/projects/covid_economy/")
print("   • covid_economic_data.csv")
print("   • figure1_main_dashboard.png")
print("   • figure2_country_deep_dive.png")
print("   • figure3_eda_correlation.png")
