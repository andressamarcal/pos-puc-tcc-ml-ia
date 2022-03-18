import base64
import codecs
import json
import logging
import os
import shutil
import time
import warnings
from datetime import date
from distutils.core import setup

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycaret
import pycaret.classification as pcc
import pycaret.regression as pcr
import scikitplot as skplt
import scipy.stats as stats
import seaborn as sns
import shap
import sqlalchemy as sqla
import streamlit as st
import streamlit.components.v1 as components
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from pandas_profiling import ProfileReport
from plot_metric.functions import BinaryClassification
from pycaret.utils import check_metric
from sklearn import metrics
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from streamlit import caching

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AutoML App",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.set_option("deprecation.showPyplotGlobalUse", False)
st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.header("AutoML App\n\n\n")
st.sidebar.image("../img/ml3.png")


@st.cache(ttl=3600, suppress_st_warning=True, allow_output_mutation=True)
def upload_data(file):
    try:
        df = pd.read_csv(file,  quotechar='"', encoding="utf-8")
        # df = pd.read_csv(file, encoding="utf-8")
        # df = pd.read_csv(file, delimiter=";", decimal=",", encoding="utf-8")
    except:
        df = pd.read_excel(file,  quotechar='"', encoding="utf-8")
    return df


# TARGET, target = None, None
# dict_values_not_tuning, dict_values_tuning = {}, {}
# df_abt, df_oot = pd.DataFrame(), pd.DataFrame()


def main():
    global dict_values_not_tuning, dict_values_tuning, df_abt, df_oot, target, TARGET

    # EXECUTION = st.sidebar.radio("Executar a Aplicação", ("Automatica", "Customizada"))
    
    TYPE = st.sidebar.selectbox(
        label="Modelagem", options=["Regressão", "Classificação"]
    )
    st.header(TYPE)
    st.markdown("___")

    if TYPE == "":
        st.stop()
    
    # if TYPE == "Tutorial para Uso":
    filedoc = codecs.open("..//markdowns//documentation.md", "r", "utf-8")
    st.write("\n\n")
    st.markdown(filedoc.read(), unsafe_allow_html=True)

    upload_file_abt = st.sidebar.file_uploader(
        label="Upload BASE TREINO", type=["csv", "xlsx"]
    )
    
    upload_file_oot = st.sidebar.file_uploader(
        label="Upload BASE TESTE", type=["csv", "xlsx"]
    )

    df_abt = upload_data(upload_file_abt) if upload_file_abt else st.stop()
    df_oot = upload_data(upload_file_oot) if upload_file_oot else None

    st.markdown("**Dataset Treino**")
    st.write(df_abt.head(20).head(20).style.highlight_null(null_color="yellow"))

    st.write("_______")
    st.markdown("**Target**")

    col1, _ = st.columns(2)

    # try:
    with col1:
        TARGET = st.selectbox(
            label="Escolha a variavel TARGET",
            options=[" "] + df_abt.columns.tolist(),
            index=(0),
        )
        target = df_abt[TARGET]
        st.bar_chart(df_abt[TARGET].value_counts())

    neg, pos = np.bincount(df_abt[TARGET])
    total = neg + pos
    st.text(
        "Instâncias:\n\n Geral: {}\n 1: {} ({:.2f}% total)\n 0: {} ({:.2f}% total)\n  ".format(
            total, pos, 100 * pos / total, neg, 100 * neg / total
        )
    )
    st.text(f"Linhas x Colunas: {df_abt.shape}")
# except:
    st.warning("**Campo Obrigatorio!**")

    # random = RandomUnderSampler(random_state=123)

    st.markdown("_______")
    st.markdown("**Comparar Modelos**")

    # if TYPE == "Classificação" and EXECUTION == "Automatica":
    if TYPE == "Classificação":
        start_button = st.button("Iniciar")
        if start_button:
            with st.spinner("Treinando Modelos"):
                # try:
                pcc.setup(
                    df_abt=df_abt,
                    TARGET=target,
                    folds=5,
                    session_id = 123
                )
                # pcc.add_metric('profit', 'Profit', calculate_profit)
                BEST = pcc.compare_models(fold=5, sort="auc")
                st.write(pcc.get_config("display_container")[1])
                st.write(BEST)
                            
                with st.expander(label = "Informações do Setup"):
                    st.write("Setup")
                    st.write(pcc.get_config("display_container")[0])
                
                st.success("Etapa concluida com sucesso!")
                
                # except:
                #     st.error("Selecione a variavel **TARGET**")
        
        st.markdown("_______")
        st.markdown("**Treinar Modelo [BASE TREINO]**")

        col5, col6, col7, col08, col09  = st.columns(5)

        try:
            build_model = pcc.create_model(
                estimator=BEST,
                fold=5,
                round=4,
                cross_validation=True,
                verbose=True,
                system=True,
            )
            st.write(build_model)

            holdout_model1 = pcc.pull()
            st.write(holdout_model1)            

            dict_values_not_tuning["model_auc"] = holdout_model1["AUC"].filter(
                like="Mean", axis=0
            )
            dict_values_not_tuning["model_sd"] = holdout_model1["AUC"].filter(
                like="SD", axis=0
            )

            with col5:
                feature_graph = pcc.plot_model(
                    build_model,
                    plot="feature",
                    save=True,
                    display_format="streamlit",
                )
                shutil.copy("Feature Importance.png", "FI_train.png")
                st.image("FI_train.png")

            with col6:
                auc_graph = pcc.plot_model(
                    build_model, plot="auc", save=True, display_format="streamlit"
                )
                shutil.copy("AUC.png", "AUC_train.png")
                st.image("AUC_train.png")

            with col7:
                confusion_matrix_graph = pcc.plot_model(
                    build_model,
                    plot="confusion_matrix",
                    save=True,
                    display_format="streamlit",
                )
                shutil.copy("Confusion Matrix.png", "CM_train.png")
                st.image("CM_train.png")

            with col08:
                ks_graph = pcc.plot_model(
                    build_model,
                    plot="ks",
                    save=True,
                    display_format="streamlit",
                )
                shutil.copy("KS Statistic Plot.png", "KS_train.png")
                st.image("KS_train.png")
            
            with col09:
                gain_graph = pcc.plot_model(
                    build_model,
                    plot="threshold",
                    save=True,
                    display_format="streamlit",
                )
                shutil.copy("Threshold.png", "Threshold_train.png")
                st.image("Threshold_train.png")
            
            # with col010:
            #     gain_graph = pcc.plot_model(
            #         build_model,
            #         plot="error",
            #         save=True,
            #         display_format="streamlit",
            #     )
            #     shutil.copy("Prediction Error.png", "Prediction Error_train.png")
            #     st.image("Prediction Error_train.png")
                
                
            # with col011:
            #     gain_graph = pcc.plot_model(
            #         build_model,
            #         plot="learning",
            #         save=True,
            #         display_format="streamlit",
            #     )
            #     shutil.copy("Learning Curve.png", "Learning Curve_train.png")
            #     st.image("Learning Curve_train.png")
            
            st.success("Modelo Construido!")
            
        except Exception as e:
            # st.error(e)
            print("ERRO: ",e)

        st.markdown("_______")
        st.markdown("**Tuning do Modelo**")

        col8, col9, col10, col11, col12 = st.columns(5)

        try:
            model_tuned = pcc.tune_model(build_model, fold=5, n_iter=30, optimize="AUC")
            st.write(model_tuned)

            holdout_tune_model = pcc.pull()
            st.write(holdout_tune_model)

            dict_values_tuning["tune_model_auc"] = holdout_tune_model["AUC"].filter(
                like="Mean", axis=0
            )

            dict_values_tuning["tune_model_sd"] = holdout_tune_model["AUC"].filter(
                like="SD", axis=0
            )

            with col8:
                feature_graph_tuned = pcc.plot_model(
                    model_tuned,
                    plot="feature",
                    save=True,
                    display_format="streamlit",
                )

                shutil.copy("Feature Importance.png", "FI_tuned.png")
                st.image("FI_tuned.png")

            with col9:
                auc_graph_tuned = pcc.plot_model(
                    model_tuned, plot="auc", save=True, display_format="streamlit"
                )

                shutil.copy("AUC.png", "AUC_tuned.png")
                st.image("AUC_tuned.png")

            with col10:
                confusion_matrix_graph_tuned = pcc.plot_model(
                    model_tuned,
                    plot="confusion_matrix",
                    save=True,
                    display_format="streamlit",
                )

                shutil.copy("Confusion Matrix.png", "CM_tuned.png")
                st.image("CM_tuned.png")

            with col11:
                ks_graph_tuned = pcc.plot_model(
                    model_tuned,
                    plot="ks",
                    save=True,
                    display_format="streamlit",
                )
                shutil.copy("KS Statistic Plot.png", "KS_tuned.png")
                st.image("KS_tuned.png")
                
            with col12:
                gain_graph_tuned = pcc.plot_model(
                    model_tuned,
                    plot="threshold",
                    save=True,
                    display_format="streamlit",
                )
                shutil.copy("Threshold.png", "Threshold_tuned.png")
                st.image("Threshold_tuned.png")
            
            # with col13:
            #     gain_graph = pcc.plot_model(
            #         model_tuned,
            #         plot="error",
            #         save=True,
            #         display_format="streamlit",
            #     )
            #     shutil.copy("Prediction Error.png", "Prediction Error_tuned.png")
            #     st.image("Prediction Error_tuned.png")
                
                
            # with col14:
            #     gain_graph = pcc.plot_model(
            #         model_tuned,
            #         plot="learning",
            #         save=True,
            #         display_format="streamlit",
            #     )
            #     shutil.copy("Learning Curve.png", "Learning Curve_tuned.png")
            #     st.image("Learning Curve_tuned.png")
            
            st.success("Tunning Concluido!")

        except UnboundLocalError as e:
            print("ERRO: ", e)

        st.markdown("_______")
        st.markdown("**Interpretabilidade [BASE TREINO]**")
        st.markdown(
            "**Atenção:** Essa funcionalidade só pode ser usada após a etapa de _Construção do Modelo_"
        )

        st.warning(
            "Usar essa opção apenas para os algoritmos **rf, et, catboost, lightgbm, dt, xgboost.**"
        )

        col111, _ = st.columns(2)

        try:
            if (
                dict_values_not_tuning["model_auc"].values
                >= dict_values_tuning["tune_model_auc"].values
                and dict_values_not_tuning["model_sd"].values
                > dict_values_tuning["tune_model_sd"].values
            ):
                with col111:
                    model_interpreted = pcc.interpret_model(build_model)
                    st.pyplot()                    
                # with col121:
                #     model_interpreted2 = pcc.interpret_model(build_model, plot="correlation")
                #     st.pyplot()
                #     st.success("Etapa concluida!")

            else:
                with col111:
                    model_interpreted = pcc.interpret_model(model_tuned)
                    st.pyplot()
                # with col121:
                #     model_interpreted2 = pcc.interpret_model(model_tuned, plot="correlation")
                #     st.pyplot()
                #     st.success("Etapa concluida!")
        
        except Exception as e:
            print("ERRO: ", e)

        st.markdown("_______")
        st.markdown("**Avaliação e Previsões do modelo [BASE TREINO]**")

        try:
            if (
                dict_values_not_tuning["model_auc"].values
                >= dict_values_tuning["tune_model_auc"].values
                and dict_values_not_tuning["model_sd"].values
                > dict_values_tuning["tune_model_sd"].values
            ):
                predictions = pcc.predict_model(build_model)
                st.write(predictions.head())
            else:
                predictions = pcc.predict_model(model_tuned)
                st.write(predictions.head())
           
        except Exception as error:
            print("ERRO: ",error)
        
        st.markdown("_______")
        st.markdown("**Exportar Arquivo Final**")

        try:
            if (
                dict_values_not_tuning["model_auc"].values
                >= dict_values_tuning["tune_model_auc"].values
                and dict_values_not_tuning["model_sd"].values
                > dict_values_tuning["tune_model_sd"].values
            ):
                model_name = "portocred_automl_{}".format(date.today())
                final_model = pcc.finalize_model(build_model)
                saved = pcc.save_model(final_model, model_name=model_name)
                st.write(saved)

                if saved:
                    st.success("Arquivo .pkl gerado com sucesso!")

            else:
                model_name = "portocred_automl_{}".format(date.today())
                final_model = pcc.finalize_model(model_tuned)
                saved = pcc.save_model(final_model, model_name=model_name)
                st.write(saved)

                if saved:
                    st.success("Arquivo .pkl gerado com sucesso!")
                    
            
            st.markdown("_______")
            st.markdown("** Grupos Homogeneos [TREINO]**")
            
            col22, col23 = st.columns(2)
            
            predictions["proba"] = predictions.apply(lambda x: x["Score"] if x["Label"] == 1 else 1-x["Score"], axis=1)

            d1 = pd.DataFrame(
                {'Bucket': pd.qcut(predictions['proba'], 10), 'Num': 1})
                    
            d4 = d1.groupby(["Bucket"], as_index=False)["Num"].count().reset_index()
            d4['Bucket'] = d4['Bucket'].astype(str)
            d4 = d4.drop(['index'], axis=1)

            l4, l5 = [], []

            for j, i in enumerate(d4['Bucket'], 1):
                i = i.replace('(', '')
                i = i.replace(']', '')
                l5.append('GH ' + str(j))
                for x  in i.split(','):
                    l4.append(float(x))

            st.markdown("_______")
                    
            predictions['GH']= pd.cut(predictions.proba, bins = sorted(set(l4)), labels=l5)
            predictions[TARGET] = predictions[TARGET].astype(int)
                 
            qtd = predictions.groupby(["GH"])[TARGET].count()
            prc = round(predictions.groupby(["GH"])[TARGET].sum()/predictions.groupby(['GH'])[TARGET].count(), 2)

            ghs = pd.DataFrame([], columns=["Volumetria", "%"])
            ghs["Volumetria"], ghs["%"] = qtd, prc

            with col22:
                st.dataframe(d4) 
            with col23:
                st.dataframe(ghs.reset_index().astype(str))
            
            matplotlib.rc_file_defaults()
            ax1 = sns.set_style(style=None, rc=None )
            fig, ax1 = plt.subplots(figsize=(16,3))

            sns.lineplot(data = ghs['%'], marker='o', sort = False, ax=ax1)
            ax2 = ax1.twinx()
            sns.barplot(data = ghs, x=l5, y='Volumetria', alpha=0.5, ax=ax2)

            st.pyplot()

        except Exception as e:
            # st.error(e)
            print("Erro: ",e)

        st.markdown("_______")
        st.markdown("## Avaliando Base OOT")

        try:
            loaded_dt_model = pcc.load_model(model_name)
            predictions_oot = pcc.predict_model(loaded_dt_model, data=df_oot)

            st.markdown("_______")
            st.markdown("**Predições OOT**")
            st.write(predictions_oot.head())
         
            st.markdown("_______")
            st.markdown("**Métricas OOT**")
            results = pcc.pull()
            st.write(results)
            
            try:
                st.markdown("_______")
                st.markdown("**KS e GINI da Base OOT**")
                
                predictions_oot["proba"] = predictions_oot.apply(lambda x: x["Score"] if x["Label"] == 1 else 1-x["Score"], axis=1)
                ks_stat = stats.ks_2samp(predictions_oot.loc[predictions_oot[TARGET]==0,"proba"], predictions_oot.loc[predictions_oot[TARGET]==1,"proba"])            
                calc_gini = (results['AUC']*2)-1    
                    
                st.markdown("\n* KS: %.2f" % round(ks_stat[0], 2))
                st.markdown("* GINI: %.2f" % round(calc_gini, 2)) 

            except Exception as e:
                st.error("Variavel target não está disponivel nos seus dados: ", e)
            
            X_train = pcc.get_config('X_train')
            X_test = pcc.get_config('X_test')
            y_train = pcc.get_config('y_train')
            y_test = pcc.get_config('y_test')
            
            st.markdown("_______")
            st.markdown("** Grupos Homogeneos [OOT]**")            
            
            predictions_oot["proba"] = predictions_oot.apply(lambda x: x["Score"] if x["Label"] == 1 else 1-x["Score"], axis=1)
            
            col20, col21 = st.columns(2)
            
            d1 = pd.DataFrame(
                {'Bucket': pd.qcut(predictions_oot['proba'], 10), 'Num': 1})
                    
            d2 = d1.groupby(["Bucket"], as_index=False)["Num"].count().reset_index()
            d2[['Bucket']] = d2[['Bucket']].astype(str)
            d2 = d2.drop(['index'], axis=1)

            l2, l3 = [], []

            for j, i in enumerate(d2['Bucket'], 1):
                i = i.replace('(', '')
                i = i.replace(']', '')
                l3.append('GH ' + str(j))
                for x  in i.split(','):
                    l2.append(float(x))

            st.markdown("_______")
                    
            predictions_oot['GH']= pd.cut(predictions_oot.proba, bins = sorted(set(l2)), labels=l3)
                        
            qtd = predictions_oot.groupby(["GH"])[TARGET].count()
            prc = predictions_oot.groupby(["GH"])[TARGET].sum()/predictions_oot.groupby(['GH'])[TARGET].count()

            ghs = pd.DataFrame([], columns=["Volumetria", "%"])
            ghs["Volumetria"], ghs["%"] = qtd, prc

            with col20:
                st.dataframe(d2) 
            with col21:
                st.dataframe(ghs.reset_index().astype(str))
            
            
            matplotlib.rc_file_defaults()
            ax1 = sns.set_style(style=None, rc=None )
            fig, ax1 = plt.subplots(figsize=(16,3))

            sns.lineplot(data = ghs['%'], marker='o', sort = False, ax=ax1)
            ax2 = ax1.twinx()
            sns.barplot(data = ghs, x=l3, y='Volumetria', alpha=0.5, ax=ax2)

            st.pyplot()
            
            st.markdown("_______")
            st.markdown("** PSI **")
            
            d4['proporcao-treino'] = d4['Num']/d4['Num'].sum()
            d2['proporcao-oot'] = d2['Num']/d2['Num'].sum()
            d2['proporcao-treino']= d4['proporcao-treino']

            d2['PSI'] = (d4['proporcao-treino'] - d2['proporcao-oot']) * np.log(d4['proporcao-treino'] / d2['proporcao-oot'])
            d4['PSI'] = (d4['proporcao-treino'] - d2['proporcao-oot']) * np.log(d4['proporcao-treino'] / d2['proporcao-oot'])
            
            st.write(d2)
            
            st.markdown("_______")
            st.markdown("**Ponto de Corte**")      
            
            numbers = [float(x)/10 for x in range(10)]
            df4 = predictions.copy()
            y_train_pred_final = []
            
            for i in numbers:
                df4[i]= predictions['proba'].map(lambda x: 1 if x > i else 0)

            cutoff_df = pd.DataFrame(columns = ['prob','accuracy','sensi','speci'])
            
            num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            for i in num:
                cm1 = metrics.confusion_matrix(df4[TARGET], df4[i])
                total1=sum(sum(cm1))
                accuracy = (cm1[0,0]+cm1[1,1])/total1                
                speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
                sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
                cutoff_df.loc[i] =[i,accuracy,sensi,speci]
            
            st.write(cutoff_df)
            
            col30, col31 = st.columns(2)

            with col30:
                cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
                plt.show()
                st.pyplot()
            
            bc = BinaryClassification(y_test, df4['Label'], labels=["Class 1", "Class 2"])
            
            with col31:
                plt.figure(figsize=(5,5))
                bc.plot_roc_curve()
                plt.show()
                st.pyplot()          

        except Exception as e:
            # st.error(e)
            print("ERRO: ", e) 
        
        #clear dictionary
        dict_values_not_tuning.clear()
        dict_values_tuning.clear()

    # elif TYPE == "Classificação" and EXECUTION == "Customizada":
    #     customize()

    # elif TYPE == "Regressão":
    #     regression_pycaret()


if __name__ == "__main__":
    main()
