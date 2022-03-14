import base64
import codecs
import json
import logging
import os
import time
import shutil
from datetime import date

# plotting
import matplotlib.pyplot as plt
import numpy as np

# Standart
import pandas as pd
import pycaret
import pycaret.classification as pcc
import pycaret.regression as pcr
import seaborn as sns
import shap

# sql
import sqlalchemy as sqla

# Streamlit
import streamlit as st
import streamlit.components.v1 as components
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Pandas Profiling
from pandas_profiling import ProfileReport

# ml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from streamlit import caching

# import SessionState
# from regression import regression_pycaret
import warnings

warnings.filterwarnings("ignore")


@st.cache(ttl=3600, suppress_st_warning=True, allow_output_mutation=True)
def upload_data(file):
    try:
        # df = pd.read_csv(file, encoding="utf-8")
        df = pd.read_csv(file, delimiter=";", decimal=",", encoding="utf-8")
    except:
        df = pd.read_excel(file)
    return df


def select_best_model(key: str):
    model_selected = st.selectbox(
        label="Selecione o modelo que deseja usar",
        options=["", "Modelo 1 (SEM TUNING)", "Modelo 2 (COM TUNING)"],
        key=key,
    )
    return model_selected


@st.cache(ttl=3600, suppress_st_warning=True, allow_output_mutation=True)
def setup_classification(
    df_abt,
    key_vars=None,
    num_vars=None,
    cat_vars=None,
    ordinal_vars=None,
    date_vars=None,
    TARGET=None,
    folds=5,
    outliers=False,
    data_normalize=False,
    random=False,
):

    SETUPCLASSIFICATION = pcc.setup(
        data=df_abt,
        # test_data=df_oot if df_oot else False,
        target=TARGET,
        train_size=0.8,
        ignore_features=key_vars if key_vars else None,
        categorical_features=cat_vars if cat_vars else None,
        numeric_features=num_vars if num_vars else None,
        ordinal_features=ordinal_vars if ordinal_vars else None,
        date_features=date_vars if date_vars else None,
        remove_outliers=outliers if outliers else False,
        normalize=True,
        normalize_method=data_normalize if data_normalize else False,
        fix_imbalance=True if random else False,
        fix_imbalance_method=random,
        silent=True,
        session_id=123,
        html=False,
    )

    BEST = pcc.compare_models(fold=folds, sort="auc")
    st.write(pcc.get_config("display_container")[1])
    st.write(BEST)

    return SETUPCLASSIFICATION


# @st.cache(ttl=3600, suppress_st_warning=True, allow_output_mutation=True)
# def step_tunning_model(build_model, folds=5, interations=30):
#     col8, col9, col10 = st.beta_columns(3)

#     try:
#         with col8:
#             feature_graph_tuned = pcc.plot_model(
#                 model_tuned,
#                 plot="feature",
#                 save=True,
#                 display_format="streamlit",
#             )

#             shutil.copy("Feature Importance.png", "FI_tuned.png")
#             st.image("FI_tuned.png")

#         with col9:
#             auc_graph_tuned = pcc.plot_model(
#                 model_tuned, plot="auc", save=True, display_format="streamlit"
#             )

#             shutil.copy("AUC.png", "AUC_tuned.png")
#             st.image("AUC_tuned.png")

#         with col10:
#             confusion_matrix_graph_tuned = pcc.plot_model(
#                 model_tuned,
#                 plot="confusion_matrix",
#                 save=True,
#                 display_format="streamlit",
#             )

#             shutil.copy("Confusion Matrix.png", "CM_tuned.png")
#             st.image("CM_tuned.png")

#         st.success("Tunning Concluido!")

#     except:
#         st.warning("Ops, algo deu errado!")


@st.cache(ttl=3600, suppress_st_warning=True, allow_output_mutation=True)
def step_interpretation_model(select_interpretation, build_model, model_tuned):
    col11, _ = st.beta_columns(2)

    try:
        if select_interpretation == "Modelo 1 (SEM TUNING)":
            with col11:
                model_interpreted = pcc.interpret_model(build_model, plot="summary")
                st.pyplot()
                st.success("Etapa concluida!")

        elif select_interpretation == "Modelo 2 (COM TUNING)":
            with col11:
                model_interpreted = pcc.interpret_model(model_tuned, plot="summary")
                st.pyplot()
                st.success("Etapa concluida!")
    except:
        st.error("Algo deu errado.")

    return model_interpreted


@st.cache(ttl=3600, suppress_st_warning=True, allow_output_mutation=True)
def avaliation_model(build_model, model_tuned, select_predict):

    if select_predict == "Modelo 1 (SEM TUNING)":
        predictions = pcc.predict_model(build_model)
        st.write(predictions)

    elif select_predict == "Modelo 2 (COM TUNING)":
        predictions = pcc.predict_model(model_tuned)
        st.write(predictions)

    return predictions


@st.cache(ttl=3600, suppress_st_warning=True, allow_output_mutation=True)
def step_one_oot(model_name, df_oot):

    try:
        loaded_dt_model = pcc.load_model(model_name)
    except:
        st.info("Salve o .pkl antes de executar essa etapa.")

    if st.button("Predict OOT"):
        predictions_oot = pcc.predict_model(loaded_dt_model, data=df_oot)

    st.markdown("Predições OOT")
    st.write(predictions_oot)

    return predictions_oot


@st.cache(ttl=3600, suppress_st_warning=True, allow_output_mutation=True)
def step_two_oot(predictions, TARGET):

    acc = accuracy_score(predictions_oot.TARGET, predictions["Label"])
    auc = roc_auc_score(predictions_oot.TARGET, predictions["Score"])
    recall = recall_score(predictions_oot.TARGET, predictions["Label"])
    precision = precision_score(predictions_oot.TARGET, predictions["Label"])
    f1 = f1_score(predictions_oot.TARGET, predictions_oot["Label"])

    cols = ["Accuracy", "AUC", "Recall", "Prec.", "F1"]
    values = [acc, auc, recall, precision, f1]
    metrics_oot = pd.DataFrame({tup[0]: [tup[1]] for tup in zip(cols, values)})

    st.markdown("Métricas Base OOT")
    st.dataframe(metrics_oot)

    st.makrdown("write")
    st.write(metrics_oot)

    return metrics_oot

    # utils
    # finished = pcc.pull()
    # automl = pcc.automl(optimize="AUC", use_holdout=True)

    # algorithm_options = {
    #     "Nome": [
    #         "Logistic Regression",
    #         "K Nearest Neighbour",
    #         "Naives Bayes",
    #         "Decision Tree Classifier",
    #         "SVM – Linear Kernel",
    #         "SVM – Radial Kernel",
    #         "Gaussian Process Classifier",
    #         "Multi Level Perceptron",
    #         "Ridge Classifier",
    #         "Random Forest Classifier",
    #         "Quadratic Discriminant Analysis",
    #         "Ada Boost Classifier",
    #         "Gradient Boosting Classifier",
    #         "Linear Discriminant Analysis",
    #         "Extra Trees Classifier",
    #         "Extreme Gradient Boosting",
    #         "Light Gradient Boosting",
    #         "CatBoost Classifier",
    #     ],
    #     "Sigla": [
    #         "lr",
    #         "knn",
    #         "nb",
    #         "dt",
    #         "svm",
    #         "rbfsvm",
    #         "gpc",
    #         "mlp",
    #         "ridge",
    #         "rf",
    #         "qda",
    #         "ada",
    #         "gbc",
    #         "lda",
    #         "et",
    #         "xgboost",
    #         "lightgbm",
    #         "catboost",
    #     ],
    # }

    # df_algoritms = pd.DataFrame(algorithm_options)

    # components.html(
    #     html=df_algoritms.to_html(index=False),
    #     scrolling=True,
    #     height=1000,
    # )


def customize():

    TYPE = st.sidebar.selectbox(
        label="Modelagem", options=["", "Classificação", "Regressão"]
    )
    st.header(TYPE)
    st.markdown("___")

    if TYPE == "":
        st.stop()

    upload_file_abt = st.sidebar.file_uploader(
        label="Upload Base TREINO", type=["csv", "xlsx"]
    )
    # upload_file_oot = st.sidebar.file_uploader(
    #     label="Upload Base OOT", type=["csv", "xlsx"]
    # )

    if upload_file_abt:
        df_abt = upload_data(upload_file_abt)
    # elif upload_file_oot:
    #     df_oot = upload_data(upload_file_oot)
    else:
        st.stop()

    st.markdown("**Dataset Treino**")
    st.write(df_abt.head(50).head(50).style.highlight_null(null_color="yellow"))

    # report_file = st.checkbox(label="Exibir detalhes dos dados", value=False)
    # st.warning("**Atenção!** Esta opção leva muito tempo para ser carregada.")

    # if report_file:
    #     with st.beta_expander("Report Base {}").format(df_abt):
    #         components.html(
    #             html=ProfileReport(df=df_abt, minimal=True).to_html(),
    #             scrolling=True,
    #             height=1000,
    # )

    st.markdown("_______")
    st.markdown("**Seleção de Variáveis**")

    key_vars = st.multiselect(
        label="Variaveis Chaves",
        options=[" "] + df_abt.columns.tolist(),
    )

    num_vars = st.multiselect(
        label="Variaveis Numericas",
        options=[" "] + df_abt.columns.tolist(),
    )

    cat_vars = st.multiselect(
        label="Variaveis Categoricas",
        options=[" "] + df_abt.columns.tolist(),
    )

    ordinal_vars = st.multiselect(
        label="Variaveis Ordinais",
        options=[" "] + df_abt.columns.tolist(),
    )

    date_vars = st.multiselect(
        label="Variaveis com Datas",
        options=[" "] + df_abt.columns.tolist(),
    )

    st.write("_______")
    st.markdown("**Target**")

    col1, _ = st.beta_columns(2)

    try:
        with col1:
            TARGET = st.selectbox(
                label="Escolha a variavel TARGET",
                options=[" "] + df_abt.columns.tolist(),
                index=(0),
            )
            st.markdown("\* _Campo de preenchimento obrigatorio_")
            st.bar_chart(df_abt[TARGET].value_counts())

        neg, pos = np.bincount(df_abt[TARGET])
        total = neg + pos
        st.text(
            "Instâncias:\n\n Geral: {}\n 1: {} ({:.2f}% total)\n 0: {} ({:.2f}% total)\n  ".format(
                total, pos, 100 * pos / total, neg, 100 * neg / total
            )
        )
        st.text(f"Linhas x Colunas: {df_abt.shape}")
    except:
        pass

    col1, col2, col3, col4 = st.beta_columns(4)

    with col1:
        st.markdown("_______")
        st.markdown("**Balanceamento dos Dados**")

        if st.checkbox("Undersample"):
            random_options = st.number_input("Random State", value=123)
            random = RandomUnderSampler(random_state=random_options)
            True
        else:
            random = None

        if st.checkbox("Oversample"):
            random = RandomOverSampler(sampling_strategy="minority")
            True
        else:
            random = None

    with col2:
        st.markdown("_______")
        st.markdown("**Outliers**")

        outliers = st.checkbox("Remover Outliers")
        if outliers:
            True

    with col3:
        st.markdown("_______")
        st.markdown("**Normalizar Dados**")

        data_normalize = st.selectbox(
            label="Tecnica",
            options=["zscore", "minmax", "maxabs", "robust"],
        )
        st.info("**Default:** zscore")

    with col4:
        st.markdown("_______")
        st.markdown("**KFold**")
        folds = st.number_input(
            "Quantidade de Folds", min_value=0, max_value=100, value=5
        )

    st.markdown("_______")
    st.markdown("**Comparar Modelos**")

    if TYPE == "Classificação":
        if st.button("Iniciar"):
            setup_classification(
                df_abt=df_abt,
                key_vars=key_vars,
                num_vars=num_vars,
                cat_vars=cat_vars,
                ordinal_vars=ordinal_vars,
                date_vars=date_vars,
                TARGET=TARGET,
                folds=folds,
                random=random,
                outliers=outliers,
                data_normalize=data_normalize,
            )
            st.success("Etapa concluida com sucesso!")

        st.markdown("_______")
        st.markdown("**Treinar Modelo [BASE TREINO]**")

        algorithm = st.selectbox(
            label="Selecione o Algoritmo que será usado",
            options=[
                "",
                "lr",
                "knn",
                "nb",
                "dt",
                "svm",
                "rbfsvm",
                "gpc",
                "mlp",
                "ridge",
                "rf",
                "qda",
                "ada",
                "gbc",
                "lda",
                "et",
                "xgboost",
                "lightgbm",
                "catboost",
            ],
        )

        col5, col6, col7 = st.beta_columns(3)

        try:
            if algorithm:
                build_model = pcc.create_model(
                    estimator=algorithm,
                    fold=folds,
                    round=4,
                    cross_validation=True,
                    verbose=True,
                    system=True,
                )
                st.write(build_model)
                holdout_model1 = pcc.pull()
                st.write(holdout_model1)

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
                st.success("Modelo Construido!")

        except:
            st.error("Ops, algo deu errado!")

        st.markdown("_______")
        st.markdown("**Tuning do Modelo**")
        interations = st.number_input(
            "Nº de Interações", min_value=None, max_value=100, value=30
        )

        col8, col9, col10 = st.beta_columns(3)

        try:
            model_tuned = pcc.tune_model(
                build_model, fold=folds, n_iter=interations, optimize="AUC"
            )

            st.write(model_tuned)
            holdout_tune_model = pcc.pull()
            st.write(holdout_tune_model)

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

            st.success("Tunning Concluido!")

        except UnboundLocalError as e:
            st.info(
                "Essa etapa só pode ser executada após a construção do modelo **[Etapa Anterior]**"
            )

        st.markdown("_______")
        st.markdown("**Interpretabilidade [BASE TREINO]**")
        st.markdown(
            "**Atenção:** Essa funcionalidade só pode ser usada após a etapa de _Construção do Modelo_"
        )

        select_interpretation = select_best_model("selectbox_interpretation")
        st.warning(
            "Usar essa opção apenas para os algoritmos **rf, et, catboost, lightgbm, dt, xgboost.**"
        )
        try:
            step_interpretation_model(
                select_interpretation=select_interpretation,
                build_model=build_model,
                model_tuned=model_tuned,
            )
        except:
            # st.info("Essa etapa só pode ser executada após a construção do modelo")
            pass

        st.markdown("_______")
        st.markdown("**Avaliação e Previsões do modelo [BASE TREINO]**")

        select_predict = select_best_model("selectbox_predict")

        try:
            avaliation_model(build_model, model_tuned, select_predict)

        except:
            # st.info("Essa etapa só pode ser executada após a construção do modelo")
            pass

        st.markdown("_______")
        st.markdown("**Exportar Arquivo Final**")

        select_final_file = select_best_model("selectbox_donwload_pickel")

        if select_final_file == "Modelo 1 (SEM TUNING)":
            model_name = "portocred_automl_{}".format(date.today())
            final_model = pcc.finalize_model(build_model)
            saved = pcc.save_model(final_model, model_name=model_name)
            st.write(saved)

            if saved:
                # st.balloons()
                st.success("Arquivo .pkl gerado com sucesso!")

        elif select_predict == "Modelo 2 (COM TUNING)":
            model_name = "portocred_automl_{}".format(date.today())
            final_model = pcc.finalize_model(model_tuned)
            saved = pcc.save_model(final_model, model_name=model_name)
            st.write(saved)

            if saved:
                # st.balloons()
                st.success("Arquivo .pkl gerado com sucesso!")

        st.markdown("_______")
        st.markdown("**Avaliando Base OOT**")

        try:
            upload_file_oot = st.sidebar.file_uploader(
                label="Upload da Base OOT", type=["csv", "xlsx"]
            )
            df_oot = upload_data(upload_file_oot)

            # step_one_oot(model_name, df_oot)

            loaded_dt_model = pcc.load_model(model_name)
            # st.info("Salve o .pkl antes de executar essa etapa.")

            if st.button("Predict OOT"):
                predictions_oot = pcc.predict_model(loaded_dt_model, data=df_oot)

            st.markdown("Predições OOT")
            st.write(predictions_oot)

            acc = accuracy_score(predictions_oot.TARGET, predictions_oot["Label"])
            auc = roc_auc_score(predictions_oot.TARGET, predictions_oot["Score"])
            recall = recall_score(predictions_oot.TARGET, predictions_oot["Label"])
            precision = precision_score(
                predictions_oot.TARGET, predictions_oot["Label"]
            )
            f1 = f1_score(predictions_oot.TARGET, predictions_oot["Label"])

            cols = ["Accuracy", "AUC", "Recall", "Prec.", "F1"]
            values = [acc, auc, recall, precision, f1]
            metrics_oot = pd.DataFrame({tup[0]: [tup[1]] for tup in zip(cols, values)})

            st.markdown("Métricas Base OOT")
            st.dataframe(metrics_oot)
            st.makrdown("write")
            st.write(metrics_oot)

            # try:
            #     step_two_oot(predictions)

            # except:
            #     pass

        except Exception as e:
            print("ERROOOOO: ", e)
            st.info("Etapa disponível após gerar arquivo .pkl")