import pandas as pd
import streamlit as st
import joblib
from io import StringIO

# Configuração da página
st.set_page_config(page_title="Previsões em Planilha", layout="centered")

# Barra lateral
st.sidebar.header('Suba sua planilha')
st.sidebar.markdown("Faça upload de um arquivo CSV com os dados.")

# Título principal
st.title("Planilha de Previsões")

# Descrição
st.markdown('Para realizar previsões, suba seu arquivo no formato **CSV**.')

# Carregar o modelo
try:
    modelo = joblib.load("modelo/modelo_ml.pkl")
except FileNotFoundError:
    st.error("Erro: Arquivo do modelo não encontrado. Certifique-se de que 'modelo_ml.pkl' está no diretório correto.")
    st.stop()

# Upload do arquivo
data = st.file_uploader('Upload do arquivo CSV', type=['csv'])

if data:
    try:
        # Ler o arquivo CSV
        df_input = pd.read_csv(data)
        
        # Verificar as colunas esperadas pelo modelo
        colunas_esperadas = modelo.feature_names_in_
        if not set(colunas_esperadas).issubset(df_input.columns):
            st.error(f"Erro: O arquivo enviado está faltando colunas. O modelo espera as seguintes colunas: {list(colunas_esperadas)}")
        else:
            # Fazer a predição
            churn_predicao = modelo.predict(df_input)
            
            # Adicionar a predição ao DataFrame
            df_output = df_input.assign(prediction=["Churn" if pred == 1 else "Não Churn" for pred in churn_predicao])
            
            # Exibir os resultados
            st.markdown("### Previsões de Churn")
            st.write(df_output)
            
            # Botão para download do arquivo com as previsões
            st.download_button(
                label='Baixar Previsões em CSV',
                data=df_output.to_csv(index=False).encode('utf-8'),
                mime='text/csv',
                file_name='churn_predictions.csv'
            )
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
