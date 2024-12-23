import pandas as pd
import streamlit as st
import joblib
from datetime import timedelta

# Configuração da página
st.set_page_config(page_title="Predição", layout="centered")

# Barra lateral
st.sidebar.header('Previsão de Churn')
st.sidebar.write("Prencha os valores para obter a previsão")

# Título principal
st.title("Predição de Indicadores")

# Input para idade
num__Idade = st.number_input(
    label='Idade', 
    value=18, 
    min_value=18, 
    max_value=120
)

# Input para tempo na plataforma (em dias ou horas)
tempo_opcao = st.radio(
    "Selecione a unidade para o Tempo na Plataforma:",
    ("Dias", "Horas")
)

if tempo_opcao == "Dias":
    tempo_na_plataforma = st.number_input(
        label="Tempo na Plataforma (em dias)",
        value=1.0,
        min_value=0.0,
        step=1.0
    )
    # Converte de dias para meses
    tempo_na_plataforma_em_meses = tempo_na_plataforma / 30.4375
else:
    tempo_na_plataforma = st.number_input(
        label="Tempo na Plataforma (em horas)",
        value=1.0,
        min_value=0.0,
        step=1.0
    )
    # Converte de horas para meses
    tempo_na_plataforma_em_meses = (tempo_na_plataforma / 24) / 30.4375

# Input para nota média
num__Nota_Média = st.number_input(
    label='Nota Média', 
    value=0, 
    min_value=0, 
    max_value=10, 
    step=0, 
    
)

# Input para tempo por usuário ativo
num__tempo_por_usuario_ativo = st.number_input(
    label='Tempo por Usuário Ativo (em horas)', 
    value=1.0, 
    min_value=0.0, 
    max_value=24.0, 
    step=0.1
)

# Conversão do tempo por usuário ativo para hh:mm:ss
total_segundos = int(num__tempo_por_usuario_ativo * 3600)
tempo_formatado = str(timedelta(seconds=total_segundos))

# Exibindo os valores inseridos
st.write("### Valores Inseridos")
st.write(f"**Idade**: {num__Idade} anos")
st.write(f"**Tempo na Plataforma**: {tempo_na_plataforma} ({tempo_opcao}) ≈ {tempo_na_plataforma_em_meses:.2f} meses")
st.write(f"**Nota Média**: {num__Nota_Média}")
st.write(f"**Tempo formatado**: {tempo_formatado} (hh:mm:ss)")

# Botão para fazer a predição
if st.button("Fazer Predição"):
    try:
        # Carregar o modelo salvo
        modelo = joblib.load("modelo/modelo_ml.pkl")

        # Criar um DataFrame com os inputs
        dados_usuario = pd.DataFrame({
            "Idade": [num__Idade],
            "Tempo_na_Plataforma": [tempo_na_plataforma_em_meses],
            "Nota_Média": [num__Nota_Média],
            "tempo_por_usuario_ativo": [num__tempo_por_usuario_ativo]
        })

        # Fazer a predição
        predicao = modelo.predict(dados_usuario)

        # Mapear o resultado para Churn ou Não Churn
        resultado = "Churn" if predicao[0] == 1 else "Não Churn"

        # Exibir o resultado formatado
        st.success(f"Predição realizada com sucesso: **{resultado}**")

    except FileNotFoundError:
        st.error("Erro: Arquivo do modelo não encontrado. Certifique-se de que 'modelo_ml.pkl' está no diretório correto.")
    except Exception as e:
        st.error(f"Erro durante a predição: {e}")

