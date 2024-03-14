import streamlit as st
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

def run():
    # Título de la aplicación
    st.title('Modelo de Supervivencia para Préstamos')

    # Carga de datos
    st.header('Carga de Datos')
    uploaded_file = st.file_uploader("Elige un archivo de Excel", type=['xlsx'])
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.write(data.head())  # Mostrar las primeras filas de los datos

        # Asegurando que 'PorcentajeDesembolsado' esté en formato adecuado y creando 'Evento'
        data['Evento'] = (data['PorcentajeDesembolsado'] < 1).astype(int)

        # Preparar el DataFrame para el modelo, excluyendo 'NoOperacion'
        data_prepared = data[['Meses', 'Evento']].copy()
        data_prepared = pd.concat([data_prepared, pd.get_dummies(data[['Sector', 'SubSectorNombre','Pais']], drop_first=True)], axis=1)

        # Asegurarse de que 'Años' es numérico (no debería haber errores de tipo si 'Años' ya es numérico)
        data_prepared['Meses'] = pd.to_numeric(data_prepared['Meses'], errors='coerce')
        
        # Normalizar las variables numéricas (excepto la columna 'Evento')
        scaler = StandardScaler()
        columns_to_scale = ['Meses']  # Añade aquí otras columnas numéricas si las hay
        data_prepared[columns_to_scale] = scaler.fit_transform(data_prepared[columns_to_scale])
        
        # Ajustar el modelo de supervivencia con regularización
        cox_model = CoxPHFitter(penalizer=0.1)  # Ajusta el valor del penalizador según sea necesario
        cox_model.fit(data_prepared, duration_col='Meses', event_col='Evento')

        # Mostrar el resumen del modelo
        st.write(cox_model.summary)

        import matplotlib.pyplot as plt
        from lifelines import KaplanMeierFitter

        # Asumiendo que 'Tiempo' es la duración hasta el evento y 'Evento' es si se alcanzó el 100% del desembolso (1 si sí, 0 si no)
        kmf = KaplanMeierFitter()

        # Ajusta el modelo de Kaplan-Meier a los datos
        kmf.fit(data['Meses'], event_observed=data['Evento'])

        # Crear y mostrar la curva de supervivencia Kaplan-Meier
        kmf = KaplanMeierFitter()
        kmf.fit(data['Años'], event_observed=data['Evento'])

        # Crea la figura para el gráfico
        fig, ax = plt.subplots()
        kmf.plot(ax=ax)
        ax.set_title('Curva de Supervivencia de Kaplan-Meier')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Probabilidad de no haber alcanzado el 100% del desembolso')
        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

if __name__ == "__main__":
    run()
