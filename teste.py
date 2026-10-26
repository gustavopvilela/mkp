import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Ajuste o caminho caso o seu CSV esteja dentro da pasta 'graficos_slides'
df = pd.read_csv('graficos_slides/Teste_Wilcoxon_Slide15.csv')

# Limpa a extensão '.dat' do nome das instâncias para o gráfico ficar mais limpo
df['Instancia'] = df['Instancia'].str.replace('.dat', '', regex=False)

fig, ax = plt.subplots(figsize=(8, len(df) * 0.5 + 2))
ax.set_xlim(-2, 3)
ax.set_ylim(-len(df) - 1, 1)
ax.axis('off')

cmap_blue = plt.get_cmap('Blues')
color_sig = '#A7F0C7'  # Verde claro

ax.text(0, 0, 'BCS', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(1.1, 0, 'BFA', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(2.2, 0, 'p-valor', ha='center', va='center', fontsize=12, fontweight='bold')

for i, row in df.iterrows():
    y = -i - 1
    inst = row['Instancia']
    bcs_val = row['Vitórias_BCS(%)']
    bfa_val = row['Vitórias_BFA(%)']

    ax.text(-0.6, y, inst, ha='right', va='center', fontsize=11)

    bcs_color_intensity = (bcs_val / 100.0) * 0.7 + 0.1
    rect_bcs = Rectangle((-0.5, y - 0.45), 1, 0.9, facecolor=cmap_blue(bcs_color_intensity), edgecolor='white')
    ax.add_patch(rect_bcs)
    text_color_bcs = 'white' if bcs_val > 50 else 'black'
    ax.text(0, y, f"{bcs_val:.0f}%", ha='center', va='center', color=text_color_bcs, fontsize=11)

    bfa_color_intensity = (bfa_val / 100.0) * 0.7 + 0.1
    rect_bfa = Rectangle((0.6, y - 0.45), 1, 0.9, facecolor=cmap_blue(bfa_color_intensity), edgecolor='white')
    ax.add_patch(rect_bfa)
    text_color_bfa = 'white' if bfa_val > 50 else 'black'
    ax.text(1.1, y, f"{bfa_val:.0f}%", ha='center', va='center', color=text_color_bfa, fontsize=11)

    rect_pval = Rectangle((1.7, y - 0.45), 1, 0.9, facecolor=color_sig, edgecolor='white')
    ax.add_patch(rect_pval)
    ax.text(2.2, y, "< 0,001 *", ha='center', va='center', color='black', fontsize=11, fontweight='bold')

plt.title('Wilcoxon pareado: % de vitórias por instância (* = p < 0.05)', pad=20, fontweight='bold')
plt.tight_layout()

plt.savefig("Slide15_Tabela_Wilcoxon.png", dpi=300, bbox_inches='tight')
print("Imagem salva com sucesso!")