import os
import time
import math
from numba import njit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
from scipy.special import gamma
from scipy.stats import wilcoxon


# =============================================================================
# 1. LEITURA E PRÉ-PROCESSAMENTO (Conforme Seção 4.1)
# =============================================================================
def read_mkp_instance(filepath):
    with open(filepath, 'r') as f:
        tokens = f.read().split()

    if not tokens:
        return None

    n = int(tokens[0])
    m = int(tokens[1])
    ref_value = float(tokens[2])

    idx = 3
    p = np.array([float(x) for x in tokens[idx: idx + n]])
    idx += n

    W_flat = [float(x) for x in tokens[idx: idx + (m * n)]]
    W = np.array(W_flat).reshape((m, n))
    idx += (m * n)

    c = np.array([float(x) for x in tokens[idx: idx + m]])

    return p, W, c, ref_value, n, m


def preprocess_instance(p, W, c):
    n = len(p)
    # Relaxação linear (0 <= x <= 1) com método HiGHS
    res = linprog(-p, A_ub=W, b_ub=c, bounds=(0, 1), method='highs')

    # Extração dos valores duais
    duals = np.abs(res.ineqlin.marginals)

    # Cálculo das pseudoutilidades
    u = np.zeros(n)
    for j in range(n):
        den = np.sum(duals * W[:, j])
        if den > 0:
            u[j] = p[j] / den
        else:
            u[j] = np.inf  # Denominador zero (itens que não consomem recursos críticos)

    # Ordem de remoção (DROP) crescente e inserção (ADD) decrescente
    order_drop = np.lexsort((p, u))
    order_add = order_drop[::-1]

    return res.fun * -1, order_drop, order_add


# =============================================================================
# 2. OPERADOR DE REPARO (Chu-Beasley)
# =============================================================================
@njit
def repair(x_in, p, W, c, order_drop, order_add):
    x = x_in.copy()

    m = len(c)
    n = len(x)

    # 1. Substitui o np.dot() problemático por um loop manual rápido
    consumo = np.zeros(m)
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += W[i, j] * x[j]
        consumo[i] = s

    # Fase DROP
    for j in order_drop:
        is_over = False
        for i in range(m):
            if consumo[i] > c[i]:
                is_over = True
                break

        if is_over:
            if x[j] == 1:
                x[j] = 0
                # Substitui consumo -= W[:, j]
                for i in range(m):
                    consumo[i] -= W[i, j]
        else:
            break

    # Fase ADD
    for j in order_add:
        if x[j] == 0:
            is_under = True
            for i in range(m):
                if consumo[i] + W[i, j] > c[i]:
                    is_under = False
                    break
            if is_under:
                x[j] = 1
                # Substitui consumo += W[:, j]
                for i in range(m):
                    consumo[i] += W[i, j]

    fit = 0.0
    for j in range(n):
        if x[j] == 1:
            fit += p[j]

    return x, fit

# =============================================================================
# 3. BUSCA DO CUCO BINÁRIA (BCS)
# =============================================================================
def levy_flight(n, beta=1.5, alpha=1.0):
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (num / den) ** (1 / beta)

    u = np.random.normal(0, sigma_u, n)
    v = np.random.normal(0, 1, n)
    step = u / (np.abs(v) ** (1 / beta))
    return alpha * step


def binary_cuckoo_search(p, W, c, order_drop, order_add, max_evals, N=25, pa=0.25, beta_levy=1.5):
    n = len(p)
    alpha = 1.0 / n
    evals = 0
    history = []

    nests = [np.random.randint(0, 2, n) for _ in range(N)]
    repaired = [repair(x, p, W, c, order_drop, order_add) for x in nests]
    nests = [r[0] for r in repaired]
    fitness = [r[1] for r in repaired]
    evals += N

    best_idx = np.argmax(fitness)
    best_nest = nests[best_idx].copy()
    best_fit = fitness[best_idx]
    history.append((evals, best_fit))

    while evals < max_evals:
        for i in range(N):
            if evals >= max_evals: break

            s = levy_flight(n, beta_levy, alpha)
            prob = np.abs(np.tanh(s))  # Função V

            flips = np.random.rand(n) < prob
            v_nest = np.where(flips, 1 - nests[i], nests[i])

            v_nest, v_fit = repair(v_nest, p, W, c, order_drop, order_add)
            evals += 1

            k = np.random.randint(0, N)
            if v_fit > fitness[k]:
                nests[k] = v_nest
                fitness[k] = v_fit

            if v_fit > best_fit:
                best_fit = v_fit
                best_nest = v_nest.copy()
                history.append((evals, best_fit))

        # Abandono
        if evals < max_evals:
            worst_indices = np.argsort(fitness)[:math.ceil(pa * N)]
            for idx in worst_indices:
                if evals >= max_evals: break
                if idx != best_idx:
                    rand_nest = np.random.randint(0, 2, n)
                    nests[idx], fitness[idx] = repair(rand_nest, p, W, c, order_drop, order_add)
                    evals += 1
                    if fitness[idx] > best_fit:
                        best_fit = fitness[idx]
                        best_nest = nests[idx].copy()
                        history.append((evals, best_fit))

    history.append((max_evals, best_fit))
    return best_nest, best_fit, history


# =============================================================================
# 4. ALGORITMO DO VAGA-LUME BINÁRIO (BFA)
# =============================================================================
def binary_firefly_algorithm(p, W, c, order_drop, order_add, max_evals, N=25, beta0=1.0, gamma_param=1.0, theta=0.98):
    n = len(p)
    alpha = 1.0 / n
    evals = 0
    history = []

    fireflies = [np.random.randint(0, 2, n) for _ in range(N)]
    repaired = [repair(x, p, W, c, order_drop, order_add) for x in fireflies]
    fireflies = [r[0] for r in repaired]
    fitness = [r[1] for r in repaired]
    evals += N

    best_idx = np.argmax(fitness)
    best_firefly = fireflies[best_idx].copy()
    best_fit = fitness[best_idx]
    history.append((evals, best_fit))

    while evals < max_evals:
        for i in range(N):
            for j in range(N):
                if evals >= max_evals: break
                if fitness[j] > fitness[i]:
                    r = np.sum(fireflies[i] != fireflies[j]) / n
                    beta = beta0 * np.exp(-gamma_param * (r ** 2))

                    diff_mask = fireflies[i] != fireflies[j]
                    attract_flips = np.random.rand(n) < beta
                    v_firefly = np.where(diff_mask & attract_flips, fireflies[j], fireflies[i])

                    rand_flips = np.random.rand(n) < alpha
                    v_firefly = np.where(rand_flips, 1 - v_firefly, v_firefly)

                    fireflies[i], fit_val = repair(v_firefly, p, W, c, order_drop, order_add)
                    evals += 1

                    if fit_val > fitness[i]:
                        fitness[i] = fit_val
                    if fit_val > best_fit:
                        best_fit = fit_val
                        best_firefly = fireflies[i].copy()
                        history.append((evals, best_fit))

        if evals < max_evals:
            b = np.argmax(fitness)
            rand_flips_b = np.random.rand(n) < alpha
            v_b = np.where(rand_flips_b, 1 - fireflies[b], fireflies[b])
            v_b, fit_b = repair(v_b, p, W, c, order_drop, order_add)
            evals += 1

            if fit_b > fitness[b]:
                fireflies[b] = v_b
                fitness[b] = fit_b
            if fit_b > best_fit:
                best_fit = fit_b
                best_firefly = v_b.copy()
                history.append((evals, best_fit))

        alpha *= theta

    history.append((max_evals, best_fit))
    return best_firefly, best_fit, history


# =============================================================================
# 5. GERADOR DE GRÁFICOS (Para os Slides)
# =============================================================================
def plot_results_for_slides(df, conv_data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Cores padronizadas
    paleta = {'BCS': '#5C82C6', 'BFA': '#E58561'}

    # 1. Slide 7: Tempo de Execução por Instância
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Instance', y='Time (s)', hue='Method', palette=paleta, capsize=.1, errorbar='sd')
    plt.title("Tempo de Execução por Instância (média e desvio)")
    plt.ylabel("Tempo (s)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Slide07_TempoExecucao.png", dpi=300)
    plt.close()

    # 2. Slide 9: Boxplot Geral dos Gaps
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Method', y='Gap (%)', hue='Method', palette=paleta, legend=False)
    plt.title("Distribuição do Gap por Método (Todas as Instâncias)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Slide09_BoxplotGeral.png", dpi=300)
    plt.close()

    # 3. Slide 10: Boxplot Gaps BCS
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[df['Method'] == 'BCS'], x='Instance', y='Gap (%)', color=paleta['BCS'])
    plt.title("BCS: Distribuição do Gap por Instância")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Slide10_BoxplotBCS.png", dpi=300)
    plt.close()

    # 4. Slide 11: Boxplot Gaps BFA
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[df['Method'] == 'BFA'], x='Instance', y='Gap (%)', color=paleta['BFA'])
    plt.title("BFA: Distribuição do Gap por Instância")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Slide11_BoxplotBFA.png", dpi=300)
    plt.close()

    # 5. Slide 14: Intervalo de Confiança (IC 95%)
    plt.figure(figsize=(12, 6))
    sns.pointplot(data=df, x='Instance', y='Gap (%)', hue='Method', palette=paleta, errorbar=('ci', 95), capsize=.2, linestyle='none', dodge=0.4)
    plt.title("Gap Médio (%) com IC de 95%")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Slide14_IntervaloConfianca.png", dpi=300)
    plt.close()

    # 6. Slides 12 e 13: Curvas de Convergência (Pequena e Grande)
    for inst, data in conv_data.items():
        plt.figure(figsize=(10, 5))

        evals_bcs, fits_bcs = zip(*data['BCS'])
        evals_bfa, fits_bfa = zip(*data['BFA'])

        # Cria degraus (step plot) para visualização correta da convergência
        plt.step(evals_bcs, fits_bcs, label='BCS', color=paleta['BCS'], where='post', linewidth=2)
        plt.step(evals_bfa, fits_bfa, label='BFA', color=paleta['BFA'], where='post', linewidth=2)

        plt.title(f"Convergência - Instância: {inst}")
        plt.xlabel("Avaliações da Função Objetivo")
        plt.ylabel("Melhor Lucro Encontrado")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/Slide_Convergencia_{inst}.png", dpi=300)
        plt.close()


# =============================================================================
# 6. TESTE DE WILCOXON E RELATÓRIO
# =============================================================================
def generate_stats_report(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    report = []

    # Pivotando para alinhar BCS e BFA por Semente e Instância
    df_pivot = df.pivot_table(index=['Instance', 'Seed'], columns='Method', values='Gap (%)').reset_index()

    for inst in df_pivot['Instance'].unique():
        data_inst = df_pivot[df_pivot['Instance'] == inst]
        gap_bcs = data_inst['BCS'].values
        gap_bfa = data_inst['BFA'].values

        # Evita erro se as arrays forem idênticas (zeros)
        if np.all(gap_bcs == gap_bfa):
            p_val = 1.0
        else:
            _, p_val = wilcoxon(gap_bcs, gap_bfa)

        win_bcs = np.sum(gap_bcs < gap_bfa) / len(gap_bcs) * 100
        win_bfa = np.sum(gap_bfa < gap_bcs) / len(gap_bcs) * 100

        report.append({
            'Instancia': inst,
            'Vitórias_BCS(%)': win_bcs,
            'Vitórias_BFA(%)': win_bfa,
            'p-valor': p_val,
            'Significativo (p<0.05)': p_val < 0.05
        })

    df_report = pd.DataFrame(report)
    df_report.to_csv(f"{out_dir}/Teste_Wilcoxon_Slide15.csv", index=False)
    print("\n=== RESULTADO DO TESTE DE WILCOXON (Slide 15) ===")
    print(df_report.to_string(index=False))


# =============================================================================
# 7. FUNÇÃO PRINCIPAL (Para rodar no seu computador)
# =============================================================================
def main():
    pasta_instancias = "./instancias"  # <-- COLOQUE SUAS INSTÂNCIAS AQUI
    pasta_graficos = "./graficos_slides"

    if not os.path.exists(pasta_instancias):
        print(f"ATENÇÃO: Crie a pasta '{pasta_instancias}' e coloque os arquivos .dat e .txt lá.")
        return

    resultados = []
    dados_convergencia = {}

    # Para o teste, define o orçamento de avaliações (pode ajustar, ex: 100 * n)
    # 30 execuções independentes (seeds)
    num_seeds = 30

    arquivos = [f for f in os.listdir(pasta_instancias) if os.path.isfile(os.path.join(pasta_instancias, f))]

    for arquivo in arquivos:
        # Ignora arquivos agregados do OR-Library
        if arquivo.startswith("OR") and arquivo.endswith(".dat") and "_" not in arquivo:
            continue

        print(f"\nProcessando {arquivo}...")
        filepath = os.path.join(pasta_instancias, arquivo)
        p, W, c, ref_value, n, m = read_mkp_instance(filepath)

        # Pré-processamento (Relaxação e Duais)
        f_lp, order_drop, order_add = preprocess_instance(p, W, c)

        # Define valor ótimo/referência para o Gap
        target_value = ref_value if ref_value > 0 else f_lp

        # Orçamento justo: N avaliações baseado no tamanho do problema
        max_evals = 100 * n

        best_hist_bcs_geral = None
        best_hist_bfa_geral = None
        best_bcs_val = -1
        best_bfa_val = -1

        for seed in range(num_seeds):
            print(f"  -> Iniciando seed {seed + 1}/{num_seeds}...")
            np.random.seed(seed)

            # Executa BCS
            t0 = time.time()
            sol_bcs, fit_bcs, hist_bcs = binary_cuckoo_search(p, W, c, order_drop, order_add, max_evals)
            t_bcs = time.time() - t0
            gap_bcs = ((target_value - fit_bcs) / target_value) * 100

            resultados.append(
                {'Instance': arquivo, 'Method': 'BCS', 'Seed': seed, 'Time (s)': t_bcs, 'Fitness': fit_bcs,
                 'Gap (%)': gap_bcs})

            # Guarda a curva do melhor para plotar na convergência
            if fit_bcs > best_bcs_val:
                best_bcs_val = fit_bcs
                best_hist_bcs_geral = hist_bcs

            # Executa BFA
            t0 = time.time()
            sol_bfa, fit_bfa, hist_bfa = binary_firefly_algorithm(p, W, c, order_drop, order_add, max_evals)
            t_bfa = time.time() - t0
            gap_bfa = ((target_value - fit_bfa) / target_value) * 100

            resultados.append(
                {'Instance': arquivo, 'Method': 'BFA', 'Seed': seed, 'Time (s)': t_bfa, 'Fitness': fit_bfa,
                 'Gap (%)': gap_bfa})

            if fit_bfa > best_bfa_val:
                best_bfa_val = fit_bfa
                best_hist_bfa_geral = hist_bfa

            print(f"     Tempo BCS: {t_bcs:.2f}s | Tempo BFA: {t_bfa:.2f}s")

        # Salva convergências da melhor rodada dessa instância
        dados_convergencia[arquivo] = {'BCS': best_hist_bcs_geral, 'BFA': best_hist_bfa_geral}

    if not resultados:
        print("Nenhuma instância foi processada.")
        return

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(f"{pasta_graficos}/Resultados_Completos.csv", index=False)

    print("\nGerando gráficos para os slides...")
    plot_results_for_slides(df_resultados, dados_convergencia, pasta_graficos)
    generate_stats_report(df_resultados, pasta_graficos)
    print(f"\nTudo concluído! Gráficos salvos na pasta: {pasta_graficos}")


if __name__ == '__main__':
    main()
    pass