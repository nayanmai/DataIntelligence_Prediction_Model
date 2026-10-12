import base64
import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt


class Mmultiple_Regression:
    def __init__ (self , Data_path , target_column):

        self.Data = Data_path
        self.target_column = target_column


# FUNTION 2 FOR CLEANING AND CALCULATION OF SLOPS

    def fit(self):

        Data = self.Data.copy()
        print(Data.columns)
        Data = Data.drop(['title','Brand me','price_per_sqft'],axis=1,errors='ignore')
        Data = Data.dropna()
        print(Data.shape)


        self.X = Data.drop(self.target_column,axis=1)
        self.Y = Data[self.target_column]


        # Separate the numerical and categorical columns 
        
        self.num_column = self.X.select_dtypes(include = np.number).columns
        self.cat_column = self.X.select_dtypes(exclude = np.number).columns

        # Encode categorical columns 

        # self.column = self.X.columns

        # print(self.X.head(30))

        # Train_test_split

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(
            self.X,
            self.Y.to_numpy(),
            test_size=0.20,
            random_state=42
            )

        # Create encoder

        self.encoder = OneHotEncoder(
            drop='first',
            handle_unknown='ignore',
            sparse_output=False
        )

        # Fit Encoder on Training Data 

        Train_Encoded = self.encoder.fit_transform(
            self.x_train[self.cat_column]
        )

        Test_Encoded = self.encoder.transform(               # Here we didnt use the the fit() Because it already--                                                   
            self.x_test[self.cat_column]                     # learned the patter of converting into 1,0 in privious use 
        )

        Encoded_Columns = self.encoder.get_feature_names_out()

        

        # Convert Encoded Arrays into DataFrame

        Train_Encoded = pd.DataFrame(
            Train_Encoded,
            columns=Encoded_Columns,
            index=self.x_train.index
        )

        Test_Encoded = pd.DataFrame(
            Test_Encoded,
            columns=Encoded_Columns,
            index=self.x_test.index
        )



        # select Numerical Columns 

        Train_Num = self.x_train[self.num_column]
        Test_Num = self.x_test[self.num_column]

        # combine Numerical and Categorical COlumns 0

        self.x_train = pd.concat([Train_Num , Train_Encoded],axis=1)
        self.x_test  = pd.concat([Test_Num  , Test_Encoded],axis=1)

        # convert it in Numpy Array 


        self.x_train = self.x_train.to_numpy()
        self.x_test  = self.x_test.to_numpy()



        # ADD Intercept

        X_train = np.column_stack((
            np.ones(len(self.x_train)),
            self.x_train
        )) 

        # Calculate Coefficients
        # Formula 
        # b^ = (self.X.T self.X)^-1 self.X.T y

        self.B = np.linalg.pinv(X_train) @ self.y_train

        self.intercept = self.B[0]
        self.Slop = self.B[1:]

        print("Intercept : ",self.intercept)
        print("Slopes    : ",self.Slop)

    def predict(self):
        X_test = np.column_stack((
            np.ones(len(self.x_test)),
            self.x_test
        ))

        self.Y_pred = X_test @ self.B

        self.residuals = self.y_test - self.Y_pred

        mse = np.mean(self.residuals**2)
        rmse = np.sqrt(mse)

        self.mse = mse
        self.rmse = rmse

        Compare = pd.DataFrame({
            'Actual': self.y_test[:20],
            'Predicted':self.Y_pred[:20],
            'Residuals': self.residuals[:20]
            

            })
        

        print("MSE  : ",mse)
        print("RMSE : ",rmse)
        print(Compare)
        
        negative = np.sum(self.Y_pred < 0)
        total = len(self.Y_pred)

        print("Negative Predictions :", negative)
        print("Total Predictions    :", total)
        print("Negative Percentage  :", (negative / total) * 100, "%")

        print("Numerical columns:")
        print(self.num_column)
        print("\nCategorical columns:")
        print(self.cat_column)
        print("\nTraining shape:")
        print(self.x_train.shape)
        print("\nTesting shape:")
        print(self.x_test.shape)

        print("\nTarget:")
        print(self.y_train[:20])



    def New_Prediction(self,New_Data):

        # New_Data = {}
        # print("\nEnter values for New Prediction\n")

        # # Numerical values
        # for col in self.num_column:
        #     value = float(input(f"Enter {col}: "))
        #     New_Data[col] = value

        # # Categorical values
        # for col in self.cat_column:
        #         value = input(f"Enter {col}: ")
        #         New_Data[col] = value

       
        # Convert the New Data Into Data Frame 
        New_Data = pd.DataFrame([New_Data])

        # Seperate the Numerical and Categorical Data 
        New_num = New_Data[self.num_column]
        New_cat = New_Data[self.cat_column]

        New_Encoded = self.encoder.transform(New_cat)
        Encoded_column = self.encoder.get_feature_names_out()

        # Convert encoded data into DataFrame

        New_Encoded = pd.DataFrame(
            New_Encoded,
            columns=Encoded_column
        )

        New_Data = pd.concat(
            [
            New_num.reset_index(drop=True),
            New_Encoded.reset_index(drop=True)
            ],
            axis=1
        )

        # convert it into Numpy 

        New_Data = New_Data.to_numpy()

        # Add Intercept

        New_Data = np.column_stack((
            np.ones(len(New_Data)),New_Data
        ))

        # Prediction 

        Prediction = New_Data @ self.B

        print("Predicted Price: ", Prediction[0])

        return Prediction[0]

    def Show_Graph(self):
            residuals = self.y_test - self.Y_pred
            actual_min = min(np.min(self.y_test), np.min(self.Y_pred))
            actual_max = max(np.max(self.y_test), np.max(self.Y_pred))

            fig, (fit_ax, residual_ax) = plt.subplots(
                1,
                2,
                figsize=(13, 5),
                facecolor="#06100a"
            )
            fig.suptitle(
                "MODEL PERFORMANCE // DIAGNOSTICS",
                color="#dfffe6",
                fontsize=13,
                fontweight="bold",
                fontfamily="monospace"
            )

            fit_ax.set_facecolor("#09170e")
            fit_ax.scatter(
                self.y_test,
                self.Y_pred,
                color="#5cff79",
                edgecolors="#d5ffe0",
                linewidths=0.45,
                alpha=0.82,
                s=42,
                label="Test predictions"
            )
            fit_ax.plot(
                [actual_min, actual_max],
                [actual_min, actual_max],
                color="#55dfff",
                linestyle="--",
                linewidth=1.8,
                label="Ideal fit"
            )
            fit_ax.set_title("Actual vs predicted", color="#e6ffeb", pad=12)
            fit_ax.set_xlabel("Actual values", color="#bed8c4")
            fit_ax.set_ylabel("Predicted values", color="#bed8c4")
            fit_ax.legend(facecolor="#0b1b11", edgecolor="#244a2e", labelcolor="#e6ffeb")

            residual_ax.set_facecolor("#09170e")
            residual_ax.hist(
                residuals,
                bins=24,
                density=True,
                color="#35df75",
                edgecolor="#78ff9d",
                alpha=0.34,
                label="Residual frequency"
            )
            if np.std(residuals) > 0:
                sns.kdeplot(
                    x=residuals,
                    ax=residual_ax,
                    color="#55dfff",
                    linewidth=2.2,
                    label="Density curve"
                )
            residual_ax.axvline(
                0,
                color="#ffd166",
                linestyle="--",
                linewidth=1.7,
                label="Zero error"
            )
            residual_ax.set_title("Residual distribution", color="#e6ffeb", pad=12)
            residual_ax.set_xlabel("Prediction error", color="#bed8c4")
            residual_ax.set_ylabel("Density", color="#bed8c4")
            residual_ax.legend(facecolor="#0b1b11", edgecolor="#244a2e", labelcolor="#e6ffeb")

            for ax in (fit_ax, residual_ax):
                ax.grid(color="#295537", alpha=0.38, linestyle=":")
                ax.tick_params(colors="#bed8c4")
                for spine in ax.spines.values():
                    spine.set_color("#315c3a")

            fig.tight_layout(rect=(0, 0, 1, 0.93))
            return fig

st.set_page_config(
    page_title="Prediction Lab | Regression Console",
    page_icon=":material/terminal:",
    layout="wide"
)

with st.sidebar:
    st.subheader("Interface settings", icon=":material/tune:")
    background_file = st.file_uploader(
        "Choose the cyber background image",
        type=["png", "jpg", "jpeg", "webp"],
        key="cyber_background_image"
    )
    st.caption("Upload the background image you shared to place it behind the app.")

if background_file is not None:
    encoded_background = base64.b64encode(background_file.getvalue()).decode("ascii")
    image_mime = background_file.type or "image/png"
    background_layer = (
        f'linear-gradient(rgba(3, 9, 5, .58), rgba(3, 9, 5, .72)), '
        f'url("data:{image_mime};base64,{encoded_background}")'
    )
else:
    background_layer = (
        "radial-gradient(ellipse at 82% 8%, rgba(28, 130, 43, 0.28), transparent 34%), "
        "radial-gradient(ellipse at 8% 48%, rgba(9, 83, 28, 0.24), transparent 42%), "
        "repeating-linear-gradient(0deg, rgba(92, 255, 121, 0.035) 0px, rgba(92, 255, 121, 0.035) 1px, transparent 1px, transparent 34px), "
        "#030805"
    )

st.markdown(
    f'<style>:root {{ --cyber-background: {background_layer}; }}</style>',
    unsafe_allow_html=True
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --neon: #5cff79;
    --neon-soft: #b8ffc4;
    --panel: rgba(7, 20, 11, 0.88);
    --line: rgba(92, 255, 121, 0.38);
}
.stApp {
    background: var(--cyber-background);
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #e5ffea;
}
[data-testid="stHeader"] { background: rgba(3, 8, 5, 0.72); }
[data-testid="stToolbar"] { right: 1rem; }
.block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 4rem; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: #f1fff3 !important; }
p, label, .stCaption,
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] p { color: #d5ead9; }
[data-testid="stMarkdownContainer"] code { color: var(--neon-soft); }
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small {
    color: #e4f6e7 !important;
    opacity: 1 !important;
}
.terminal-hero {
    position: relative;
    overflow: hidden;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem 2.2rem 1.8rem;
    margin: 0.25rem 0 1.5rem;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: linear-gradient(115deg, rgba(8, 25, 13, .97), rgba(4, 12, 7, .88) 64%, rgba(18, 58, 25, .60));
    box-shadow: 0 0 42px rgba(36, 255, 75, .08), inset 0 1px rgba(255,255,255,.035);
    text-align: center;
}
.terminal-hero:after {
    content: '01 / MODEL SYSTEM';
    position: absolute;
    right: 2rem;
    bottom: 1.35rem;
    color: rgba(119, 255, 143, .42);
    font: 600 .72rem 'IBM Plex Mono', monospace;
    letter-spacing: .14em;
}
.eyebrow { color: var(--neon); font: 600 .76rem 'IBM Plex Mono', monospace; letter-spacing: .18em; }
.hero-title { margin: .55rem 0 .35rem; color: #f0fff2; font: 700 clamp(2rem, 5vw, 3.4rem) 'Space Grotesk', sans-serif; letter-spacing: -.045em; }
.hero-title span { color: var(--neon); text-shadow: 0 0 22px rgba(92,255,121,.4); }
.hero-copy { margin: 0 auto; max-width: 720px; color: #d0e5d4; font: .98rem 'Space Grotesk', sans-serif; }
.system-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: 0 0 1.7rem; }
.system-card { padding: .85rem 1rem; border: 1px solid var(--line); border-radius: 12px; background: rgba(5, 17, 9, .76); }
.system-card b { display: block; margin-bottom: .25rem; color: var(--neon); font: 600 .7rem 'IBM Plex Mono', monospace; letter-spacing: .12em; }
.system-card span { color: #e0f1e3; font: .88rem 'Space Grotesk', sans-serif; }
[data-testid="stFileUploaderDropzone"] { border: 1px dashed rgba(92,255,121,.62); border-radius: 12px; background: rgba(14, 38, 20, .72); }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--neon); background: rgba(18, 56, 27, .55); }
[data-testid="stFileUploaderDropzone"] button {
    border: 1px solid #a4ffb4 !important;
    border-radius: 8px !important;
    background: linear-gradient(120deg, #00d878, #5cff79) !important;
    color: #031006 !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
[data-testid="stFileUploaderDropzone"] button * {
    color: #031006 !important;
    fill: #031006 !important;
    opacity: 1 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(120deg, #58ffa0, #b0ff82) !important;
    box-shadow: 0 0 20px rgba(92,255,121,.34);
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stNumberInput"] [data-baseweb="input"] > div {
    border: 1px solid rgba(92,255,121,.48) !important;
    background: #0b1b10 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] * { color: #f0fff2 !important; }
[role="listbox"] { border: 1px solid var(--line); background: #09140c !important; }
[role="option"] { color: #e8f8eb !important; }
[role="option"]:hover { background: #173a20 !important; }
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primary"] {
    min-height: 3.2rem;
    padding: 0 1.5rem;
    border: 1px solid #a4ffb4 !important;
    border-radius: 10px;
    background: linear-gradient(120deg, #00d878, #5cff79) !important;
    color: #031006 !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .95rem;
    font-weight: 700;
    letter-spacing: .035em;
    opacity: 1 !important;
    box-shadow: 0 0 22px rgba(37, 255, 105, .22), inset 0 1px rgba(255,255,255,.28);
    transition: .18s ease;
}
.stButton > button[kind="primary"] *,
[data-testid="stBaseButton-primary"] *,
button[data-testid="stBaseButton-primary"] * {
    color: #031006 !important;
    fill: #031006 !important;
    opacity: 1 !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    border-color: #d1ffda !important;
    background: linear-gradient(120deg, #58ffa0, #b0ff82) !important;
    box-shadow: 0 0 30px rgba(92,255,121,.42);
    transform: translateY(-1px);
}
.stButton > button[kind="primary"]:disabled,
[data-testid="stBaseButton-primary"]:disabled,
button[data-testid="stBaseButton-primary"]:disabled {
    border-color: #45634b !important;
    background: #1a2b1e !important;
    color: #a9b8ac !important;
    box-shadow: none;
}
.stButton > button[kind="primary"]:disabled *,
[data-testid="stBaseButton-primary"]:disabled *,
button[data-testid="stBaseButton-primary"]:disabled * { color: #a9b8ac !important; }
[data-testid="stMetric"] { min-height: 120px; padding: 1rem 1.15rem; border: 1px solid var(--line); border-radius: 13px; background: linear-gradient(145deg, rgba(9, 27, 14, .95), rgba(5, 14, 8, .95)); }
[data-testid="stMetricLabel"] { color: #a8c5ae; font-family: 'IBM Plex Mono', monospace; }
[data-testid="stMetricValue"] { color: var(--neon-soft); font-family: 'Space Grotesk', sans-serif; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line); border-radius: 14px; background: rgba(5, 15, 8, .58); }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
@media (max-width: 700px) {
    .terminal-hero { padding: 1.4rem; }
    .terminal-hero:after { display: none; }
    .system-grid { grid-template-columns: 1fr; }
}
</style>

<section class="terminal-hero">
    <div class="eyebrow">CYBER CELL / ANALYTICS CONSOLE</div>
  <h1 class="hero-title">DATA <span>INTELLIGENCE</span></h1>
  <p class="hero-copy">Train a multiple linear regression model, inspect its diagnostics, and generate predictions from your own dataset.</p>
</section>
<div class="system-grid">
  <div class="system-card"><b>01 / DATA INTAKE</b><span>CSV dataset upload</span></div>
  <div class="system-card"><b>02 / MODEL CORE</b><span>Linear regression engine</span></div>
  <div class="system-card"><b>03 / INFERENCE</b><span>New-value prediction</span></div>
</div>
""", unsafe_allow_html=True)

upload_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
    )

if upload_file is not None:
    Data = pd.read_csv(upload_file)

    st.subheader("Dataset preview", icon=":material/table_view:")
    st.dataframe(Data.head(10), width="stretch")

    numeric_target_columns = Data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_target_columns:
         st.warning("This regression model needs a numeric target column. Check that your CSV has a numeric column, such as price.")

    tragetColumn = st.selectbox(
        "Select a numeric target column",
        numeric_target_columns or ["No numeric columns available"],
        disabled=not numeric_target_columns
    )

    if st.button(
         "Initialize model",
         icon=":material/rocket_launch:",
            type='primary',
            disabled=not numeric_target_columns
         ):
            models = Mmultiple_Regression(Data,tragetColumn)
            models.fit()
            models.predict()

            st.session_state.models = models
            st.session_state.trained = True

            st.success("Models Trained Successfully ! ")

    if st.session_state.get("trained", False):
         models = st.session_state.models

         st.subheader("Model diagnostics", icon=":material/query_stats:")
         col1 , col2 = st.columns(2)


         with col1:
              st.metric(
                   "MSE",
                   f"{models.mse:,.2f}"
              )

         with col2:
              st.metric(
                   "RMSE",
                   f"{models.rmse:,.2f}"
              )


         st.subheader("Actual vs. predicted", icon=":material/compare_arrows:")

         Compare = pd.DataFrame(
              {
                     'Actual': models.y_test[:20],
                     'Predicted':models.Y_pred[:20],
                     'Residuals': models.residuals[:20]
         
              }
            )
         st.dataframe(
              Compare,
              use_container_width=True
         )

         st.subheader("Residual distribution", icon=":material/monitoring:")
         fig = models.Show_Graph()
         st.pyplot(fig,)

    if not st.session_state.get("trained", False):
         st.stop()

    st.subheader(f"Generate a prediction for {tragetColumn}", icon=":material/online_prediction:")
    st.write(
         "#### Enter the Value of New Data below :-"
    )

    New_Data = {}

    for col in models.num_column:
         New_Data[col] = st.number_input(
              f"Enter : {col}",
              value=0.0
         )
    st.write(
         "#### Enter The Value Categorical Value :-"
    )
    for col in models.cat_column:

         index = list(
              models.cat_column
        ).index(col)
         
         categories = (
              models.encoder
              .categories_[index]
              .tolist()
        )
         New_Data[col] = st.selectbox(
              f"Select {col}",
              categories
        )

    if st.button(" Predict House Price"):
         
         prediction = models.New_Prediction(
              New_Data
         )


         st.success(
              f"Predicted {tragetColumn}:{prediction:,.2f}"
         )











 
