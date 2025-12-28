# app.py
import streamlit as st
import pandas as pd
import re
import os
import json
import tempfile
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="ANN File Comparator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ANNComparator:
    """
    A tool to compare two ANN (Brat annotation) files and calculate evaluation metrics
    """
    
    def __init__(self):
        self.entity_types = set()
        self.gold_file = ""
        self.pred_file = ""
        
    def parse_ann_file(self, file_content: str) -> Dict[str, List]:
        """
        Parse .ann file content and extract entities with their text
        """
        entities = []
        
        try:
            # Parse each line
            for line in file_content.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Entity lines start with T (e.g., T1\tDiagnosis 0 5\tcancer)
                if line.startswith('T'):
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        entity_id = parts[0]
                        
                        # Parse type and span
                        type_span = parts[1].split()
                        if len(type_span) >= 3:
                            entity_type = type_span[0]
                            start = int(type_span[1])
                            end = int(type_span[2])
                            text = parts[2]
                            
                            entities.append({
                                'id': entity_id,
                                'type': entity_type,
                                'text': text,
                                'start': start,
                                'end': end,
                                'full_text': line
                            })
                            
                            self.entity_types.add(entity_type)
                            
        except Exception as e:
            st.error(f"Error parsing ANN file: {e}")
            return {'entities': []}
            
        return {'entities': entities}
    
    def match_entities(self, gold_entities: List[Dict], pred_entities: List[Dict], 
                      match_type: str = 'exact') -> Tuple[List, List, List]:
        """
        Match entities between gold and prediction
        match_type: 'exact', 'partial', or 'type_only'
        Returns: (true_positives, false_positives, false_negatives)
        """
        tp = []  # True Positives
        fp = []  # False Positives
        fn = []  # False Negatives
        
        gold_matched = set()
        pred_matched = set()
        
        # First pass: Find exact or partial matches
        for i, pred_entity in enumerate(pred_entities):
            matched = False
            
            for j, gold_entity in enumerate(gold_entities):
                if match_type == 'exact':
                    # Exact match: same type and exact span
                    if (pred_entity['type'] == gold_entity['type'] and
                        pred_entity['start'] == gold_entity['start'] and
                        pred_entity['end'] == gold_entity['end']):
                        
                        tp.append({
                            'gold': gold_entity,
                            'pred': pred_entity,
                            'match_type': 'exact',
                            'status': 'MATCHED'
                        })
                        gold_matched.add(j)
                        pred_matched.add(i)
                        matched = True
                        break
                        
                elif match_type == 'partial':
                    # Partial match: same type and overlapping span
                    if pred_entity['type'] == gold_entity['type']:
                        # Calculate overlap
                        overlap_start = max(pred_entity['start'], gold_entity['start'])
                        overlap_end = min(pred_entity['end'], gold_entity['end'])
                        
                        if overlap_start < overlap_end:  # There is overlap
                            overlap_length = overlap_end - overlap_start
                            pred_length = pred_entity['end'] - pred_entity['start']
                            gold_length = gold_entity['end'] - gold_entity['start']
                            
                            # Require at least 50% overlap with the smaller entity
                            min_length = min(pred_length, gold_length)
                            if overlap_length / min_length >= 0.5:
                                tp.append({
                                    'gold': gold_entity,
                                    'pred': pred_entity,
                                    'match_type': 'partial',
                                    'overlap_ratio': overlap_length / min_length,
                                    'status': 'PARTIAL_MATCH'
                                })
                                gold_matched.add(j)
                                pred_matched.add(i)
                                matched = True
                                break
                
                elif match_type == 'type_only':
                    # Type-only match: same type (ignores span)
                    if pred_entity['type'] == gold_entity['type']:
                        tp.append({
                            'gold': gold_entity,
                            'pred': pred_entity,
                            'match_type': 'type_only',
                            'status': 'TYPE_MATCH'
                        })
                        gold_matched.add(j)
                        pred_matched.add(i)
                        matched = True
                        break
            
            if not matched:
                fp.append({
                    'pred': pred_entity,
                    'status': 'FALSE_POSITIVE'
                })
        
        # Find false negatives (gold entities not matched)
        for j, gold_entity in enumerate(gold_entities):
            if j not in gold_matched:
                fn.append({
                    'gold': gold_entity,
                    'status': 'FALSE_NEGATIVE'
                })
        
        return tp, fp, fn
    
    def calculate_metrics(self, tp: List, fp: List, fn: List) -> Dict:
        """
        Calculate precision, recall, and F1 score
        """
        tp_count = len(tp)
        fp_count = len(fp)
        fn_count = len(fn)
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'TP': tp_count,
            'FP': fp_count,
            'FN': fn_count,
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4)
        }
    
    def calculate_metrics_by_type(self, gold_entities: List[Dict], pred_entities: List[Dict]) -> pd.DataFrame:
        """
        Calculate metrics for each entity type separately
        """
        results = []
        
        # Group entities by type
        gold_by_type = defaultdict(list)
        pred_by_type = defaultdict(list)
        
        for entity in gold_entities:
            gold_by_type[entity['type']].append(entity)
            
        for entity in pred_entities:
            pred_by_type[entity['type']].append(entity)
        
        # Calculate metrics for each type
        all_types = sorted(set(list(gold_by_type.keys()) + list(pred_by_type.keys())))
        
        for entity_type in all_types:
            type_gold = gold_by_type.get(entity_type, [])
            type_pred = pred_by_type.get(entity_type, [])
            
            # Use exact matching for per-type metrics
            tp, fp, fn = self.match_entities(type_gold, type_pred, 'exact')
            metrics = self.calculate_metrics(tp, fp, fn)
            
            results.append({
                'Entity Type': entity_type,
                'TP': metrics['TP'],
                'FP': metrics['FP'],
                'FN': metrics['FN'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1': metrics['F1'],
                'Gold Support': len(type_gold),
                'Pred Support': len(type_pred)
            })
        
        # Sort by entity type
        return pd.DataFrame(results).sort_values('Entity Type')
    
    def create_alignment_dataframe(self, tp: List, fp: List, fn: List) -> pd.DataFrame:
        """
        Create a DataFrame showing entity-by-entity alignment
        """
        alignment_data = []
        
        # Add True Positives (matched entities)
        for match in tp:
            alignment_data.append({
                'Status': match.get('status', 'MATCHED'),
                'Match Type': match.get('match_type', 'exact'),
                'Gold Entity ID': match['gold']['id'],
                'Pred Entity ID': match['pred']['id'],
                'Entity Type': match['gold']['type'],
                'Gold Text': match['gold']['text'],
                'Pred Text': match['pred']['text'],
                'Gold Span': f"{match['gold']['start']}-{match['gold']['end']}",
                'Pred Span': f"{match['pred']['start']}-{match['pred']['end']}",
                'Overlap Ratio': match.get('overlap_ratio', 1.0),
                'Gold Full Annotation': match['gold']['full_text'],
                'Pred Full Annotation': match['pred']['full_text']
            })
        
        # Add False Positives (predicted but not in gold)
        for mismatch in fp:
            alignment_data.append({
                'Status': 'FALSE_POSITIVE',
                'Match Type': 'NONE',
                'Gold Entity ID': 'N/A',
                'Pred Entity ID': mismatch['pred']['id'],
                'Entity Type': mismatch['pred']['type'],
                'Gold Text': 'N/A',
                'Pred Text': mismatch['pred']['text'],
                'Gold Span': 'N/A',
                'Pred Span': f"{mismatch['pred']['start']}-{mismatch['pred']['end']}",
                'Overlap Ratio': 0.0,
                'Gold Full Annotation': 'N/A',
                'Pred Full Annotation': mismatch['pred']['full_text']
            })
        
        # Add False Negatives (in gold but not predicted)
        for mismatch in fn:
            alignment_data.append({
                'Status': 'FALSE_NEGATIVE',
                'Match Type': 'NONE',
                'Gold Entity ID': mismatch['gold']['id'],
                'Pred Entity ID': 'N/A',
                'Entity Type': mismatch['gold']['type'],
                'Gold Text': mismatch['gold']['text'],
                'Pred Text': 'N/A',
                'Gold Span': f"{mismatch['gold']['start']}-{mismatch['gold']['end']}",
                'Pred Span': 'N/A',
                'Overlap Ratio': 0.0,
                'Gold Full Annotation': mismatch['gold']['full_text'],
                'Pred Full Annotation': 'N/A'
            })
        
        # Create DataFrame and sort by status
        df = pd.DataFrame(alignment_data)
        
        # Define custom sort order for Status
        status_order = ['MATCHED', 'PARTIAL_MATCH', 'TYPE_MATCH', 'FALSE_POSITIVE', 'FALSE_NEGATIVE']
        df['Status'] = pd.Categorical(df['Status'], categories=status_order, ordered=True)
        df = df.sort_values(['Status', 'Entity Type'])
        
        return df
    
    def create_metrics_dataframe(self, gold_entities: List[Dict], pred_entities: List[Dict], 
                                exact_tp: List, exact_fp: List, exact_fn: List) -> pd.DataFrame:
        """
        Create metrics DataFrame
        """
        metrics_data = []
        
        # Calculate exact match metrics
        exact_metrics = self.calculate_metrics(exact_tp, exact_fp, exact_fn)
        
        # Add overall row
        metrics_data.append({
            'Entity Type': 'OVERALL (Exact Match)',
            'TP': exact_metrics['TP'],
            'FP': exact_metrics['FP'],
            'FN': exact_metrics['FN'],
            'Precision': exact_metrics['Precision'],
            'Recall': exact_metrics['Recall'],
            'F1': exact_metrics['F1'],
            'Gold Support': len(gold_entities),
            'Pred Support': len(pred_entities)
        })
        
        # Calculate per-type metrics
        type_metrics_df = self.calculate_metrics_by_type(gold_entities, pred_entities)
        
        # Add per-type rows
        for _, row in type_metrics_df.iterrows():
            metrics_data.append({
                'Entity Type': row['Entity Type'],
                'TP': row['TP'],
                'FP': row['FP'],
                'FN': row['FN'],
                'Precision': row['Precision'],
                'Recall': row['Recall'],
                'F1': row['F1'],
                'Gold Support': row['Gold Support'],
                'Pred Support': row['Pred Support']
            })
        
        return pd.DataFrame(metrics_data)

def create_visualizations(metrics_df: pd.DataFrame, exact_metrics: Dict, partial_metrics: Dict):
    """Create visualizations for the metrics"""
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Score Overview", 
        "📈 Per-Type Metrics", 
        "🎯 Confusion Matrix", 
        "📋 Entity Distribution"
    ])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Exact Match F1", f"{exact_metrics['F1']:.3f}")
            st.metric("Precision", f"{exact_metrics['Precision']:.3f}")
        with col2:
            st.metric("Partial Match F1", f"{partial_metrics['F1']:.3f}")
            st.metric("Recall", f"{exact_metrics['Recall']:.3f}")
        with col3:
            st.metric("Exact TP/FP/FN", f"{exact_metrics['TP']}/{exact_metrics['FP']}/{exact_metrics['FN']}")
            st.metric("Partial TP/FP/FN", f"{partial_metrics['TP']}/{partial_metrics['FP']}/{partial_metrics['FN']}")
        
        # Radar chart for metrics
        fig = go.Figure()
        
        categories = ['Precision', 'Recall', 'F1']
        
        fig.add_trace(go.Scatterpolar(
            r=[exact_metrics['Precision'], exact_metrics['Recall'], exact_metrics['F1']],
            theta=categories,
            fill='toself',
            name='Exact Match'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[partial_metrics['Precision'], partial_metrics['Recall'], partial_metrics['F1']],
            theta=categories,
            fill='toself',
            name='Partial Match'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Performance Metrics Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Filter out overall row for per-type visualization
        per_type_df = metrics_df[metrics_df['Entity Type'] != 'OVERALL (Exact Match)']
        
        if not per_type_df.empty:
            # Bar chart for F1 scores by type
            fig = px.bar(
                per_type_df, 
                x='Entity Type', 
                y='F1',
                title='F1 Score by Entity Type',
                color='F1',
                color_continuous_scale='RdYlGn',
                range_color=[0, 1]
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed table
            st.dataframe(
                per_type_df[['Entity Type', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', 'Gold Support', 'Pred Support']],
                use_container_width=True
            )
        else:
            st.info("No per-type metrics available")
    
    with tab3:
        # Confusion matrix visualization
        fig = go.Figure(data=go.Heatmap(
            z=[[exact_metrics['TP'], exact_metrics['FP']],
               [exact_metrics['FN'], 0]],
            x=['Predicted Positive', 'Predicted Negative'],
            y=['Actual Positive', 'Actual Negative'],
            text=[[f"TP: {exact_metrics['TP']}", f"FP: {exact_metrics['FP']}"],
                  [f"FN: {exact_metrics['FN']}", ""]],
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues',
            showscale=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix (Exact Match)",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=400,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Entity distribution
        if 'per_type_df' in locals() and not per_type_df.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=per_type_df['Entity Type'],
                y=per_type_df['Gold Support'],
                name='Gold Entities',
                marker_color='blue'
            ))
            
            fig.add_trace(go.Bar(
                x=per_type_df['Entity Type'],
                y=per_type_df['Pred Support'],
                name='Predicted Entities',
                marker_color='orange'
            ))
            
            fig.update_layout(
                title='Entity Distribution by Type',
                xaxis_title='Entity Type',
                yaxis_title='Count',
                barmode='group',
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit app"""
    
    # App header
    st.title("📊 ANN File Comparator")
    st.markdown("""
    **Compare two .ann annotation files and calculate evaluation metrics**
    
    This tool compares gold standard annotations with predicted annotations and calculates:
    - Exact match metrics (same type & exact span)
    - Partial match metrics (same type & ≥50% overlap)
    - Per-entity type metrics
    - Detailed entity alignment
    """)
    
    # Sidebar for file upload
    st.sidebar.header("📁 File Upload")
    
    uploaded_gold = st.sidebar.file_uploader(
        "Upload GOLD standard .ann file", 
        type=['ann'],
        help="Upload the ground truth annotation file"
    )
    
    uploaded_pred = st.sidebar.file_uploader(
        "Upload PREDICTED .ann file", 
        type=['ann'],
        help="Upload the predicted annotation file"
    )
    
    # Advanced options
    st.sidebar.header("⚙️ Options")
    
    show_raw = st.sidebar.checkbox("Show raw file contents", value=False)
    download_format = st.sidebar.radio(
        "Download format",
        ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"]
    )
    
    # Main content area
    if uploaded_gold and uploaded_pred:
        # Read file contents
        gold_content = uploaded_gold.getvalue().decode('utf-8')
        pred_content = uploaded_pred.getvalue().decode('utf-8')
        
        # Create comparator instance
        comparator = ANNComparator()
        
        # Parse files
        with st.spinner("Parsing annotation files..."):
            gold_data = comparator.parse_ann_file(gold_content)
            pred_data = comparator.parse_ann_file(pred_content)
            
            gold_entities = gold_data['entities']
            pred_entities = pred_data['entities']
            
        # Display file info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Gold File**: {uploaded_gold.name}")
            st.metric("Gold Entities", len(gold_entities))
            
            if show_raw:
                with st.expander("View Gold File Content"):
                    st.code(gold_content)
        
        with col2:
            st.info(f"**Predicted File**: {uploaded_pred.name}")
            st.metric("Predicted Entities", len(pred_entities))
            
            if show_raw:
                with st.expander("View Predicted File Content"):
                    st.code(pred_content)
        
        if len(gold_entities) == 0:
            st.warning("⚠️ No entities found in gold file")
        if len(pred_entities) == 0:
            st.warning("⚠️ No entities found in prediction file")
        
        if len(gold_entities) > 0 and len(pred_entities) > 0:
            # Calculate matches
            with st.spinner("Calculating matches and metrics..."):
                # Exact matches
                exact_tp, exact_fp, exact_fn = comparator.match_entities(
                    gold_entities, pred_entities, 'exact'
                )
                exact_metrics = comparator.calculate_metrics(exact_tp, exact_fp, exact_fn)
                
                # Partial matches
                partial_tp, partial_fp, partial_fn = comparator.match_entities(
                    gold_entities, pred_entities, 'partial'
                )
                partial_metrics = comparator.calculate_metrics(partial_tp, partial_fp, partial_fn)
                
                # Create DataFrames
                metrics_df = comparator.create_metrics_dataframe(
                    gold_entities, pred_entities, exact_tp, exact_fp, exact_fn
                )
                alignment_df = comparator.create_alignment_dataframe(exact_tp, exact_fp, exact_fn)
            
            # Display results
            st.success("✅ Comparison completed!")
            
            # Visualizations
            create_visualizations(metrics_df, exact_metrics, partial_metrics)
            
            # Detailed alignment view
            st.header("🔍 Detailed Entity Alignment")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                status_filter = st.multiselect(
                    "Filter by Status",
                    options=alignment_df['Status'].unique(),
                    default=alignment_df['Status'].unique()
                )
            with col2:
                type_filter = st.multiselect(
                    "Filter by Entity Type",
                    options=alignment_df['Entity Type'].unique(),
                    default=alignment_df['Entity Type'].unique()
                )
            with col3:
                search_text = st.text_input("Search in text")
            
            # Apply filters
            filtered_df = alignment_df.copy()
            if status_filter:
                filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]
            if type_filter:
                filtered_df = filtered_df[filtered_df['Entity Type'].isin(type_filter)]
            if search_text:
                filtered_df = filtered_df[
                    filtered_df['Gold Text'].str.contains(search_text, case=False, na=False) |
                    filtered_df['Pred Text'].str.contains(search_text, case=False, na=False)
                ]
            
            # Display filtered results
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=400
            )
            
            # Download section
            st.header("📥 Download Results")
            
            # Prepare download data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"ann_comparison_{timestamp}"
            
            col1, col2, col3 = st.columns(3)
            
            if download_format == "Excel (.xlsx)":
                # Create Excel file
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                    alignment_df.to_excel(writer, sheet_name='Alignment', index=False)
                excel_buffer.seek(0)
                
                col1.download_button(
                    label="📥 Download Excel",
                    data=excel_buffer,
                    file_name=f"{base_filename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            elif download_format == "CSV (.csv)":
                col1.download_button(
                    label="📥 Download Metrics CSV",
                    data=metrics_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{base_filename}_metrics.csv",
                    mime="text/csv"
                )
                
                col2.download_button(
                    label="📥 Download Alignment CSV",
                    data=alignment_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{base_filename}_alignment.csv",
                    mime="text/csv"
                )
                
            else:  # JSON
                # Create JSON report
                report = {
                    'comparison_date': datetime.now().isoformat(),
                    'gold_file': uploaded_gold.name,
                    'pred_file': uploaded_pred.name,
                    'summary': {
                        'exact_match': exact_metrics,
                        'partial_match': partial_metrics
                    },
                    'metrics': metrics_df.to_dict('records'),
                    'alignment': alignment_df.to_dict('records')
                }
                
                col1.download_button(
                    label="📥 Download JSON Report",
                    data=json.dumps(report, indent=2, ensure_ascii=False).encode('utf-8'),
                    file_name=f"{base_filename}.json",
                    mime="application/json"
                )
            
            # Show summary statistics
            st.sidebar.header("📋 Summary")
            st.sidebar.metric("Exact Match F1", f"{exact_metrics['F1']:.4f}")
            st.sidebar.metric("Partial Match F1", f"{partial_metrics['F1']:.4f}")
            st.sidebar.metric("Total Gold Entities", len(gold_entities))
            st.sidebar.metric("Total Predicted Entities", len(pred_entities))
            
    else:
        # Show instructions when no files are uploaded
        st.info("👈 **Please upload both annotation files to begin comparison**")
        
        # Example section
        with st.expander("📝 Example ANN File Format"):
            st.code("""T1\tDiagnosis 0 5\tcancer
T2\tTreatment 10 20\tchemotherapy
T3\tMedication 25 35\taspirin""")
            
            st.markdown("""
            **ANN File Format:**
            - Each line represents one annotation
            - Format: `T[ID]\\t[TYPE] [START] [END]\\t[TEXT]`
            - `T1`: Entity ID
            - `Diagnosis`: Entity type
            - `0 5`: Start and end character offsets
            - `cancer`: The annotated text
            """)
        
        # Features list
        st.markdown("""
        ### ✨ Features
        
        - **Exact Match Evaluation**: Compare entities with same type and exact span
        - **Partial Match Evaluation**: Identify entities with ≥50% overlap
        - **Per-Type Metrics**: View performance for each entity type separately
        - **Visual Analytics**: Interactive charts and graphs
        - **Filterable Results**: Filter by status, type, or search text
        - **Multiple Export Formats**: Download results as Excel, CSV, or JSON
        
        ### 🚀 How to Use
        
        1. Upload your GOLD standard annotation file (.ann)
        2. Upload your PREDICTED annotation file (.ann)
        3. View the automated comparison results
        4. Explore visualizations and detailed alignment
        5. Download results in your preferred format
        """)

if __name__ == "__main__":
    main()
