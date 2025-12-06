"""
Monitoring dashboard using Streamlit
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json


def load_drift_report(filepath: str = "metrics/drift_report.json"):
    """Load drift detection report"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def load_monitoring_logs(filepath: str = "metrics/monitoring_logs.json"):
    """Load monitoring logs"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    st.set_page_config(page_title="Model Monitoring", page_icon="📊", layout="wide")
    
    st.title("📊 Model Monitoring Dashboard")
    st.markdown("Real-time monitoring of model performance and data drift")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "🔄 Data Drift", "📝 Prediction Logs"])
    
    # Performance Tab
    with tab1:
        st.header("Model Performance Metrics")
        
        logs = load_monitoring_logs()
        if logs:
            df = pd.DataFrame(logs)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", len(df))
            
            with col2:
                avg_conf = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
            
            with col3:
                if 'true_label' in df.columns:
                    labeled = df[df['true_label'].notna()]
                    if len(labeled) > 0:
                        accuracy = (labeled['prediction'] == labeled['true_label']).mean()
                        st.metric("Accuracy", f"{accuracy:.2%}")
            
            # Confidence distribution
            st.subheader("Confidence Distribution")
            fig = px.histogram(df, x='confidence', nbins=50, 
                             title="Prediction Confidence Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Predictions over time
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df_time = df.set_index('timestamp').resample('1H').size().reset_index(name='count')
                
                st.subheader("Predictions Over Time")
                fig = px.line(df_time, x='timestamp', y='count',
                            title="Prediction Volume Over Time")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No monitoring logs available yet")
    
    # Data Drift Tab
    with tab2:
        st.header("Data Drift Detection")
        
        drift_report = load_drift_report()
        if drift_report:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Features", drift_report['total_features'])
            
            with col2:
                st.metric("Features with Drift", drift_report['features_with_drift'])
            
            with col3:
                st.metric("Drift %", f"{drift_report['drift_percentage']:.1f}%")
            
            # Drifted features
            if drift_report['drifted_features']:
                st.warning(f"⚠️ Drift detected in: {', '.join(drift_report['drifted_features'])}")
            else:
                st.success("✅ No significant drift detected")
            
            # Detailed results
            st.subheader("Detailed Drift Analysis")
            results_df = pd.DataFrame(drift_report['detailed_results'])
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info("No drift detection report available yet")
    
    # Prediction Logs Tab
    with tab3:
        st.header("Recent Predictions")
        
        logs = load_monitoring_logs()
        if logs:
            df = pd.DataFrame(logs)
            st.dataframe(df.tail(100), use_container_width=True)
        else:
            st.info("No prediction logs available yet")


if __name__ == "__main__":
    main()
