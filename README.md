import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import os
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

# ==================== PROTEÇÃO POR SENHA ====================
SENHA = "Esm.2026"

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.set_page_config(page_title="Acesso Restrito", layout="centered")
    st.title("🔒 Acesso Restrito")
    senha_input = st.text_input("Digite a senha para acessar:", type="password")
    if senha_input == SENHA:
        st.session_state.autenticado = True
        st.rerun()
    else:
        if senha_input:
            st.error("Senha incorreta!")
        st.stop()
# ============================================================

# Configuração da página (só executa após autenticação)
st.set_page_config(page_title="REIT AI v23.0 - Macro & Portfolio", layout="wide")
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a2a, #1a0f2e); }
    .metric-macro { color: #00ff88; font-size: 24px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== DADOS MACRO (FRED API) ====================
FRED_API_KEY = os.getenv("FRED_API_KEY", "")  # opcional: coloque sua chave se quiser

def fetch_fred_series(series_id):
    if not FRED_API_KEY:
        return None
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        observations = data.get('observations', [])
        if observations:
            last = observations[-1]
            return float(last['value']) if last['value'] not in ['.', ''] else None
    except:
        pass
    return None

def get_macro_data():
    return {
        'fed_funds': fetch_fred_series('FEDFUNDS'),
        'cpi': fetch_fred_series('CPIAUCSL'),
        'unemployment': fetch_fred_series('UNRATE'),
        'gdp': fetch_fred_series('GDP'),
        'ten_year_treasury': fetch_fred_series('DGS10')
    }

# ==================== OTIMIZAÇÃO DE CARTEIRA ====================
def optimize_portfolio(tickers, lookback_years=2, risk_free_rate=0.03):
    returns_data = {}
    for t in tickers:
        df = yf.Ticker(t).history(period=f"{lookback_years}y")
        if not df.empty:
            returns_data[t] = df['Close'].pct_change().dropna()
    if len(returns_data) < 2:
        return None
    returns_df = pd.DataFrame(returns_data).dropna()
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weights_record.append(weights)
        port_return = np.sum(mean_returns * weights)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - risk_free_rate) / port_std
        results[0,i] = port_return
        results[1,i] = port_std
        results[2,i] = sharpe
    
    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]
    return {
        'weights': dict(zip(tickers, optimal_weights)),
        'return': results[0, max_sharpe_idx],
        'risk': results[1, max_sharpe_idx],
        'sharpe': results[2, max_sharpe_idx],
        'frontier': pd.DataFrame({'Return': results[0], 'Risk': results[1], 'Sharpe': results[2]})
    }

def plot_efficient_frontier(frontier, optimal_point, tickers):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frontier['Risk'], y=frontier['Return'], mode='markers',
                             marker=dict(color=frontier['Sharpe'], colorscale='RdYlGn', size=3),
                             name='Carteiras simuladas'))
    fig.add_trace(go.Scatter(x=[optimal_point['risk']], y=[optimal_point['return']], mode='markers',
                             marker=dict(size=15, color='gold', symbol='star'), name='Carteira ótima'))
    fig.update_layout(title="Fronteira Eficiente de Markowitz", xaxis_title="Risco Anual", yaxis_title="Retorno Anual", template='plotly_dark')
    return fig

# ==================== RELATÓRIO PDF ====================
class RiskReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'REIT AI - Risk Report', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_risk_report(ticker, var_95, cvar_95, beta, macro_data, portfolio_opt=None):
    pdf = RiskReportPDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Relatório de Risco - {ticker}", 0, 1)
    pdf.cell(0, 10, f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC", 0, 1)
    pdf.ln(5)
    pdf.cell(0, 10, f"VaR 95%: {var_95:.2%}", 0, 1)
    pdf.cell(0, 10, f"CVaR 95%: {cvar_95:.2%}", 0, 1)
    pdf.cell(0, 10, f"Beta: {beta:.2f}", 0, 1)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, "Indicadores Macroeconômicos:", 0, 1)
    pdf.set_font('Arial', '', 10)
    for k, v in macro_data.items():
        if v:
            pdf.cell(0, 6, f"{k}: {v:.2f}", 0, 1)
    if portfolio_opt:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 10, "Carteira Otimizada (Max Sharpe):", 0, 1)
        pdf.set_font('Arial', '', 10)
        for tick, w in portfolio_opt['weights'].items():
            pdf.cell(0, 6, f"{tick}: {w:.2%}", 0, 1)
        pdf.cell(0, 6, f"Retorno esperado: {portfolio_opt['return']:.2%}", 0, 1)
        pdf.cell(0, 6, f"Risco esperado: {portfolio_opt['risk']:.2%}", 0, 1)
        pdf.cell(0, 6, f"Sharpe Ratio: {portfolio_opt['sharpe']:.2f}", 0, 1)
    pdf.output("risk_report.pdf")
    return "risk_report.pdf"

# ==================== DASHBOARD PRINCIPAL ====================
def main():
    st.title("🏢 REIT AI v23.0 - Macro & Portfolio Optimization")
    tickers = ["VICI", "PLD", "O", "DOC", "SBRA", "AMT", "EQIX"]
    selected = st.multiselect("REITs para carteira", tickers, default=tickers[:3])
    
    macro = get_macro_data()
    
    tabs = st.tabs(["📈 Macro Dashboard", "📊 Portfolio Optimizer", "📉 Risk Analytics", "📄 Risk Report"])
    
    with tabs[0]:
        st.subheader("🌍 Indicadores Macroeconômicos (FRED)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fed Funds Rate", f"{macro.get('fed_funds', 'N/A')}" if macro.get('fed_funds') else "N/A")
        col2.metric("CPI (Inflação)", f"{macro.get('cpi', 'N/A')}" if macro.get('cpi') else "N/A")
        col3.metric("Desemprego", f"{macro.get('unemployment', 'N/A')}" if macro.get('unemployment') else "N/A")
        col4.metric("Taxa 10 anos", f"{macro.get('ten_year_treasury', 'N/A')}" if macro.get('ten_year_treasury') else "N/A")
        st.caption("Configure a chave FRED_API_KEY para dados reais (opcional).")
        
        if macro.get('fed_funds'):
            st.subheader("📉 Sensibilidade a juros")
            fig = go.Figure()
            for ticker in selected:
                info = yf.Ticker(ticker).info
                beta = info.get('beta', 1.0)
                rates = np.linspace(0, 7, 50)
                impact = 100 * (1 - beta * (rates - macro['fed_funds']) / 100)
                fig.add_trace(go.Scatter(x=rates, y=impact, mode='lines', name=ticker))
            fig.update_layout(title="Impacto estimado no preço", xaxis_title="Taxa de juros (%)", yaxis_title="Preço relativo (%)", template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("📈 Otimização de Carteira (Markowitz)")
        if len(selected) >= 2:
            with st.spinner("Otimizando..."):
                opt = optimize_portfolio(selected)
                if opt:
                    st.plotly_chart(plot_efficient_frontier(opt['frontier'], opt, selected), use_container_width=True)
                    st.subheader("🎯 Alocação ótima (máximo Sharpe)")
                    weights_df = pd.DataFrame([opt['weights']]).T.reset_index()
                    weights_df.columns = ['REIT', 'Peso']
                    weights_df['Peso %'] = weights_df['Peso'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(weights_df)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Retorno esperado", f"{opt['return']:.2%}")
                    col2.metric("Risco esperado", f"{opt['risk']:.2%}")
                    col3.metric("Sharpe Ratio", f"{opt['sharpe']:.2f}")
                    if st.button("Exportar carteira (CSV)"):
                        weights_df.to_csv("optimal_portfolio.csv", index=False)
                        with open("optimal_portfolio.csv", "rb") as f:
                            st.download_button("Baixar CSV", f, file_name="optimal_portfolio.csv")
                else:
                    st.warning("Não foi possível otimizar. Dados insuficientes.")
        else:
            st.info("Selecione pelo menos 2 REITs.")
    
    with tabs[2]:
        st.subheader("📉 Risk Analytics Avançado")
        risk_ticker = st.selectbox("REIT para análise", selected, key="risk_sel")
        if risk_ticker:
            df = yf.Ticker(risk_ticker).history(period="2y")
            returns = df['Close'].pct_change().dropna()
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            beta = yf.Ticker(risk_ticker).info.get('beta', 1.0)
            col1, col2, col3 = st.columns(3)
            col1.metric("VaR 95%", f"{var_95:.2%}")
            col2.metric("CVaR 95%", f"{cvar_95:.2%}")
            col3.metric("Beta", f"{beta:.2f}")
            if st.button("Executar Simulação Monte Carlo"):
                mu = returns.mean()
                sigma = returns.std()
                sims = 1000
                horizon = 252
                paths = np.zeros((sims, horizon))
                for i in range(sims):
                    shocks = np.random.normal(mu, sigma, horizon)
                    paths[i] = 100 * np.exp(np.cumsum(shocks))
                fig = go.Figure()
                for i in range(min(100, sims)):
                    fig.add_trace(go.Scatter(y=paths[i], mode='lines', line=dict(width=0.5), opacity=0.3, showlegend=False))
                fig.update_layout(title="Simulação de preço (252 dias)", template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.subheader("📄 Gerar Relatório de Risco (PDF)")
        report_ticker = st.selectbox("REIT", selected, key="report_sel")
        if st.button("Criar PDF"):
            df = yf.Ticker(report_ticker).history(period="2y")
            returns = df['Close'].pct_change().dropna()
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            beta = yf.Ticker(report_ticker).info.get('beta', 1.0)
            macro_data = get_macro_data()
            opt = None
            if len(selected) >= 2:
                opt = optimize_portfolio(selected)
            pdf_file = generate_risk_report(report_ticker, var_95, cvar_95, beta, macro_data, opt)
            with open(pdf_file, "rb") as f:
                st.download_button("Baixar PDF", f, file_name=f"{report_ticker}_risk_report.pdf")

if __name__ == "__main__":
    main()
