# # app.py
# import streamlit as st
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import KMeans
# import plotly.express as px

# # Set page config
# st.set_page_config(page_title="Agriculture Insights", layout="wide")

# # Title
# st.title("🌱 Agricultural Data Clustering Dashboard")

# # Sidebar - File Upload
# st.sidebar.header("Upload Your Data")

# crop_file = st.sidebar.file_uploader("Upload Crop Production CSV", type=["csv"])
# rainfall_file = st.sidebar.file_uploader("Upload Rainfall CSV", type=["csv"])

# if crop_file and rainfall_file:
#     # Read files
#     crop_df = pd.read_csv(crop_file)
#     rainfall_df = pd.read_csv(rainfall_file)

#     # Clean column names
#     crop_df.rename(columns={'State_Name': 'State', 'Crop_Year': 'Year'}, inplace=True)
#     rainfall_df.rename(columns={'Annual Rainfall': 'Rainfall'}, inplace=True)

#     # Group crop data
#     crop_summary = crop_df.groupby(['State', 'Year']).agg({
#         'Area': 'sum',
#         'Production': 'sum'
#     }).reset_index()

#     # Merge rainfall with crop data
#     merged_df = pd.merge(crop_summary, rainfall_df, on=['State', 'Year'], how='inner')

#     # Normalize data
#     scaler = MinMaxScaler()
#     merged_df[['Area', 'Production', 'Rainfall']] = scaler.fit_transform(
#         merged_df[['Area', 'Production', 'Rainfall']]
#     )

#     # Select number of clusters
#     k = st.sidebar.slider("Select number of clusters", min_value=2, max_value=6, value=3)

#     # Run KMeans clustering
#     model = KMeans(n_clusters=k, random_state=42)
#     merged_df['Cluster'] = model.fit_predict(merged_df[['Area', 'Production', 'Rainfall']])

#     # Main chart
#     st.subheader("📊 Cluster Visualization: Production vs Rainfall")
#     fig = px.scatter(
#         merged_df,
#         x='Production',
#         y='Rainfall',
#         color='Cluster',
#         size='Area',
#         hover_data=['State', 'Year'],
#         title='K-Means Clustering of Crop & Rainfall Data'
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # Cluster Summary
#     st.subheader("📋 Cluster-wise Summary")
#     summary = merged_df.groupby('Cluster')[['Area', 'Production', 'Rainfall']].mean().round(3)
#     st.dataframe(summary)

#     # Optional: Download result
#     csv = merged_df.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="📥 Download Clustered Data as CSV",
#         data=csv,
#         file_name="clustered_agriculture_data.csv",
#         mime='text/csv',
#     )

# else:
#     st.info("👈 Upload both Crop and Rainfall datasets to begin analysis.")

# import streamlit as st
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import plotly.express as px

# st.set_page_config(page_title="Agri-Sustainability Clustering App", layout="wide")
# st.title("🌾 Agri Sustainability Dashboard - KMeans Clustering")

# st.sidebar.header("Upload Your CSV Datasets")
# crop_file = st.sidebar.file_uploader("📂 Upload Crop Production CSV", type=["csv"])
# rainfall_file = st.sidebar.file_uploader("🌧️ Upload Rainfall CSV", type=["csv"])

# # Optional suicide file
# suicide_file = st.sidebar.file_uploader("☠️ Upload Farmer Suicide CSV (Optional)", type=["csv"])

# if crop_file and rainfall_file:
#     crop_df = pd.read_csv(crop_file)
#     rainfall_df = pd.read_csv(rainfall_file)

#     # --- Data Cleaning & Preprocessing ---
#     crop_df.rename(columns={'State_Name': 'State', 'Crop_Year': 'Year'}, inplace=True)
#     rainfall_df.rename(columns={'Annual Rainfall': 'Rainfall'}, inplace=True)

#     # Season filter
#     seasons = crop_df['Season'].unique().tolist()
#     selected_season = st.sidebar.selectbox("🌾 Select Crop Season", seasons)
#     season_crop_df = crop_df[crop_df['Season'] == selected_season]

#     # Group crop data
#     crop_summary = season_crop_df.groupby(['State', 'Year']).agg({'Area': 'sum', 'Production': 'sum'}).reset_index()

#     # Merge rainfall
#     merged_df = pd.merge(crop_summary, rainfall_df, on=['State', 'Year'], how='inner')

#     # Merge suicide data if uploaded
#     if suicide_file:
#         suicide_df = pd.read_csv(suicide_file)
#         suicide_df = suicide_df.rename(columns={suicide_df.columns[0]: 'State', suicide_df.columns[1]: 'Year', suicide_df.columns[-1]: 'Suicides'})
#         suicide_df = suicide_df[['State', 'Year', 'Suicides']]
#         suicide_df = suicide_df.groupby(['State', 'Year']).sum().reset_index()
#         merged_df = pd.merge(merged_df, suicide_df, on=['State', 'Year'], how='left')
#         merged_df['Suicides'].fillna(0, inplace=True)

#     # Normalization (0–10 scale)
#     scaler = MinMaxScaler(feature_range=(0, 10))
#     cols_to_norm = ['Area', 'Production', 'Rainfall'] + (['Suicides'] if 'Suicides' in merged_df.columns else [])
#     merged_df[cols_to_norm] = scaler.fit_transform(merged_df[cols_to_norm])

#     # Cluster selection
#     st.sidebar.subheader("🔢 Clustering")
#     max_k = st.sidebar.slider("Max K to Test", min_value=2, max_value=10, value=5)
#     features = merged_df[cols_to_norm]

#     best_k = 2
#     best_score = -1
#     for k in range(2, max_k + 1):
#         model = KMeans(n_clusters=k, random_state=42)
#         labels = model.fit_predict(features)
#         score = silhouette_score(features, labels)
#         if score > best_score:
#             best_k = k
#             best_score = score

#     st.sidebar.markdown(f"**✅ Best k = {best_k} (Silhouette Score = {best_score:.3f})**")
#     kmeans = KMeans(n_clusters=best_k, random_state=42)
#     merged_df['Cluster'] = kmeans.fit_predict(features)

#     # Visuals
#     st.subheader("📊 Cluster Visualization")
#     fig = px.scatter(
#         merged_df,
#         x='Production',
#         y='Rainfall',
#         color='Cluster',
#         hover_data=['State', 'Year', 'Area'] + (['Suicides'] if 'Suicides' in merged_df.columns else []),
#         title=f"Clusters by Production & Rainfall ({selected_season} Season)"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("📋 Cluster Summary Table")
#     st.dataframe(merged_df.groupby('Cluster')[cols_to_norm].mean().round(2))

#     # Optional download
#     csv = merged_df.to_csv(index=False).encode('utf-8')
#     st.download_button("📥 Download Clustered Data", data=csv, file_name="clustered_agriculture.csv")

# else:
#     st.warning("👈 Please upload both Crop and Rainfall datasets to begin.")


import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import base64

st.set_page_config(page_title="Agri-Sustainability Clustering App", layout="wide")

# Download link function
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# About section
with st.expander("ℹ️ About This App"):
    st.markdown("""
    **Agri Sustainability Clustering App** helps you analyze crop production patterns across different Indian states
    based on crop area, production, and annual rainfall. It uses **KMeans clustering** to group states by similarity,
    allowing identification of regions that might be at risk, underperforming, or showing signs of strong sustainability.

    **You can:**
    - Upload your crop and rainfall datasets
    - Choose crop season (e.g. Kharif, Rabi)
    - Automatically detect optimal number of clusters using silhouette score
    - Visualize the clusters via 3D scatter plots, pie charts, and parallel coordinates
    - Export the processed dataset with cluster labels

    This tool is ideal for students, researchers, and policymakers working in the domain of agriculture and sustainability.
    """)

st.title("🌾 Agri Sustainability Dashboard - KMeans Clustering")

st.sidebar.header("Upload Your CSV Datasets")
crop_file = st.sidebar.file_uploader("📂 Upload Crop Production CSV", type=["csv"])
rainfall_file = st.sidebar.file_uploader("🌧️ Upload Rainfall CSV", type=["csv"])

if crop_file and rainfall_file:
    crop_df = pd.read_csv(crop_file)
    rainfall_df = pd.read_csv(rainfall_file)

    crop_df.rename(columns={'State_Name': 'State', 'Crop_Year': 'Year'}, inplace=True)
    rainfall_df.rename(columns={'Annual Rainfall': 'Rainfall'}, inplace=True)

    seasons = crop_df['Season'].unique().tolist()
    selected_season = st.sidebar.selectbox("🌾 Select Crop Season", seasons)
    season_crop_df = crop_df[crop_df['Season'] == selected_season]

    crop_summary = season_crop_df.groupby(['State', 'Year']).agg({'Area': 'sum', 'Production': 'sum'}).reset_index()
    merged_df = pd.merge(crop_summary, rainfall_df, on=['State', 'Year'], how='inner')

    scaler = MinMaxScaler(feature_range=(0, 10))
    cols_to_norm = ['Area', 'Production', 'Rainfall']
    merged_df[cols_to_norm] = scaler.fit_transform(merged_df[cols_to_norm])

    st.sidebar.subheader("🔢 Clustering Settings")
    max_k = st.sidebar.slider("Max K to Test", min_value=2, max_value=10, value=5)
    features = merged_df[cols_to_norm]

    # Automatically detecting the best number of clusters using silhouette score
    best_k = 2
    best_score = -1
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(features)
        score = silhouette_score(features, labels)
        if score > best_score:
            best_k = k
            best_score = score

    st.sidebar.markdown(f"**✅ Best k = {best_k} (Silhouette Score = {best_score:.3f})**")
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    merged_df['Cluster'] = kmeans.fit_predict(features)

    # Display the dataframe with clusters
    st.header("🌐 Cluster Analysis")
    st.dataframe(merged_df)
    st.markdown(get_download_link(merged_df, "clustered_data.csv", "📥 Download Clustered Data"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Cluster Distribution (Pie chart)
        fig_pie = px.pie(merged_df, names='Cluster', title='Cluster Distribution')
        st.plotly_chart(fig_pie, use_container_width=True)

        # 3D scatter plot if there are at least 3 features
        if len(cols_to_norm) >= 3:
            fig_3d = px.scatter_3d(merged_df, x=cols_to_norm[0], y=cols_to_norm[1], z=cols_to_norm[2], color='Cluster', 
                                   title="3D Scatter Plot of Clusters")
            st.plotly_chart(fig_3d, use_container_width=True)

    with col2:
        # Parallel coordinates plot
        fig_parallel = px.parallel_coordinates(merged_df, color='Cluster', dimensions=cols_to_norm, title="Cluster Characteristics")
        st.plotly_chart(fig_parallel, use_container_width=True)

    # Cluster Statistics
    st.header("📊 Cluster Statistics")
    cluster_stats = merged_df.groupby('Cluster')[cols_to_norm].mean()
    st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))

    # Bar chart: Crop Area vs Production by State
    st.header("📊 Crop Area vs Production by State")
    crop_area_production = merged_df.groupby('State')[['Area', 'Production']].sum().reset_index()
    fig_bar = px.bar(crop_area_production, x='State', y=['Area', 'Production'], barmode='group', title="Crop Area and Production by State")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Bar chart: Average Rainfall per Cluster
    st.header("📊 Average Rainfall per Cluster")
    avg_rainfall = merged_df.groupby('Cluster')['Rainfall'].mean().reset_index()
    fig_rainfall_bar = px.bar(avg_rainfall, x='Cluster', y='Rainfall', title="Average Rainfall per Cluster", labels={'Rainfall': 'Average Rainfall (scaled)'})
    st.plotly_chart(fig_rainfall_bar, use_container_width=True)

    # Cluster Comparison by States
    st.header("📊 Cluster Comparison by States")
    cluster_state_counts = merged_df.groupby(['Cluster', 'State']).size().reset_index(name='Count')
    fig_state_cluster = px.bar(cluster_state_counts, x='State', y='Count', color='Cluster', title="Cluster Distribution Across States", barmode='stack')
    st.plotly_chart(fig_state_cluster, use_container_width=True)

    # Additional Analysis (Demographic breakdown or any additional features)
    st.header("📊 Additional Insights")
    # Assuming 'State' or any other feature could be used for demographic analysis
    if 'State' in merged_df.columns:
        fig_state = px.sunburst(merged_df, path=['Cluster', 'State'], title="State Distribution by Cluster")
        st.plotly_chart(fig_state, use_container_width=True)

else:
    st.warning("👈 Please upload both Crop and Rainfall datasets to begin.")



