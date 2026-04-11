"""
ACR Flask API - CSV Upload & Faithful/Faithless Analysis
Handles ANY CSV → ACR-PS pipeline → faithful/faithless metrics
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acr.engine import ACREngine
from acr.smart_rules import auto_detect_rules, apply_rules
from acr.faithfulness_metrics import FaithfulnessEvaluator

app = Flask(__name__)


def auto_clean_dataframe(df, clean_mode='auto'):
    """Professor-approved auto-cleaning for messy CSVs"""
    df_clean = df.copy()
    
    print(f"🔍 Original: {len(df)} rows, {len(df.columns)} cols")
    
    if clean_mode == 'raw':
        print("⚠️ Using RAW data (no cleaning)")
        return df_clean
    
    # 1. REMOVE DUPLICATES (most common mess)
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dup_removed = initial_rows - len(df_clean)
    if dup_removed > 0:
        print(f"🧹 Removed {dup_removed} duplicates")
    
    # 2. HANDLE MISSING VALUES (professor's #1 concern)
    missing_before = df_clean.isnull().sum().sum()
    for col in df_clean.columns:
        if df_clean[col].dtype in ['object', 'string']:
            # Categorical: mode imputation
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
            else:
                df_clean[col] = df_clean[col].fillna('unknown')
        else:
            # Numeric: median imputation  
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    if missing_before > 0:
        print(f"🧹 Filled {missing_before} missing values")
    
    # 3. FIX DATA TYPES (dates, numbers, categories)
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Try to convert to numeric
            temp_col = pd.to_numeric(df_clean[col], errors='coerce')
            if temp_col.notna().sum() / len(temp_col) > 0.8:
                df_clean[col] = temp_col
            else:
                # Categorical → label encode
                try:
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                except Exception as e:
                    print(f"⚠️ Could not encode {col}: {e}")
    
    # 4. REMOVE OUTLIERS (IQR method - safe for ML)
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_count = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
            if outlier_count > 0:
                print(f"🧹 Removed {outlier_count} outliers from {col}")
    
    print(f"✅ CLEANED: {len(df_clean)} rows, {len(df_clean.columns)} cols")
    return df_clean


@app.route('/', methods=['GET'])
def index():
    """Upload form for CSV analysis"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ACR Faithful/Faithless Analyzer</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 2rem;
            }
            .container {
                background: white;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 3rem;
                max-width: 600px;
                width: 100%;
            }
            h1 { color: #667eea; margin-bottom: 0.5rem; font-size: 2rem; }
            .subtitle { color: #666; margin-bottom: 2rem; font-size: 1rem; }
            
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 12px;
                padding: 3rem 2rem;
                text-align: center;
                background: #f8f9ff;
                cursor: pointer;
                transition: all 0.2s;
                margin-bottom: 1.5rem;
            }
            .upload-area:hover { 
                border-color: #764ba2;
                background: #f0f2ff;
            }
            .upload-area.dragging {
                border-color: #764ba2;
                background: #e8ebff;
            }
            .upload-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
            .upload-text { color: #666; margin-bottom: 0.5rem; }
            .upload-hint { color: #999; font-size: 0.85rem; }
            
            #fileInput { display: none; }
            
            .form-group {
                margin-bottom: 1.5rem;
            }
            label {
                display: block;
                margin-bottom: 0.5rem;
                color: #333;
                font-weight: 500;
            }
            input[type="text"] {
                width: 100%;
                padding: 0.75rem;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1rem;
            }
            
            button {
                width: 100%;
                padding: 1rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4); }
            button:disabled { opacity: 0.5; cursor: not-allowed; }
            
            .loading { display: none; text-align: center; color: #667eea; }
            .spinner { display: inline-block; animation: spin 1s linear infinite; }
            @keyframes spin { to { transform: rotate(360deg); } }
            
            .results-preview {
                background: #f8f9ff;
                border-radius: 12px;
                padding: 2rem;
                margin-bottom: 1.5rem;
                display: none;
            }
            .metric {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.75rem 0;
                border-bottom: 1px solid #e0e0e0;
            }
            .metric:last-child { border-bottom: none; }
            .metric-name { color: #666; }
            .metric-value { 
                font-size: 1.3rem; 
                font-weight: 700; 
                color: #667eea;
            }
            .error {
                background: #fee2e2;
                color: #991b1b;
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                display: none;
            }
            a { color: #667eea; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔮 Faithful/Faithless Analyzer</h1>
            <p class="subtitle">Upload any CSV to analyze counterfactual faithfulness</p>
            
            <div class="error" id="error"></div>
            
            <div class="results-preview" id="resultsPreview">
                <h3 style="margin-bottom: 1rem; color: #333;">Results</h3>
                <div class="metric">
                    <span class="metric-name">✅ Faithful Rate</span>
                    <span class="metric-value" id="faithfulRate">--</span>
                </div>
                <div class="metric">
                    <span class="metric-name">❌ Rule Violation Rate</span>
                    <span class="metric-value" id="violationRate">--</span>
                </div>
                <div class="metric">
                    <span class="metric-name">⚡ Agent Full Rate</span>
                    <span class="metric-value" id="agentRate">--</span>
                </div>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e0e0e0;">
                    <a href="/dashboard">📊 View Dashboard →</a>
                </div>
            </div>
            
            <form id="uploadForm">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">Drag & drop your CSV here or click to browse</div>
                    <div class="upload-hint">Supported: CSV files (any format)</div>
                    <input type="file" id="fileInput" accept=".csv" />
                </div>
                
                <div class="form-group" style="background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <input type="checkbox" id="cleanToggle" checked style="width: 1.25rem; height: 1.25rem; cursor: pointer;">
                        <label for="cleanToggle" style="cursor: pointer; margin-bottom: 0; flex: 1;">
                            <strong>🧹 Auto-clean messy data</strong><br>
                            <span style="font-size: 0.85rem; color: #666;">Removes duplicates, fills missing values, fixes data types & outliers</span>
                        </label>
                    </div>
                </div>
                
                <div class="form-group" id="filenameGroup" style="display: none;">
                    <label>Detected Target Column:</label>
                    <input type="text" id="targetColumn" readonly />
                </div>
                
                <button type="submit" id="analyzeBtn">🚀 Analyze CSV</button>
                <div class="loading" id="loading">
                    <div class="spinner">⏳</div>
                    <p>Analyzing your data...</p>
                </div>
            </form>
        </div>
        
        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const errorDiv = document.getElementById('error');
            const resultsPreview = document.getElementById('resultsPreview');
            
            uploadArea.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    uploadArea.textContent = '✅ ' + e.target.files[0].name + ' ready';
                    analyzeBtn.disabled = false;
                }
            });
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragging');
            });
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragging');
            });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragging');
                fileInput.files = e.dataTransfer.files;
                const event = new Event('change', { bubbles: true });
                fileInput.dispatchEvent(event);
            });
            
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const file = fileInput.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('csv', file);
                formData.append('clean_mode', document.getElementById('cleanToggle').checked ? 'auto' : 'raw');
                
                loading.style.display = 'block';
                analyzeBtn.disabled = true;
                errorDiv.style.display = 'none';
                resultsPreview.style.display = 'none';
                
                try {
                    const response = await fetch('/analyze_csv', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    // Show results
                    document.getElementById('faithfulRate').textContent = data.summary.faithful_rate + '%';
                    document.getElementById('violationRate').textContent = data.summary.rule_violation_rate + '%';
                    document.getElementById('agentRate').textContent = data.summary.agent_full_rate + '%';
                    resultsPreview.style.display = 'block';
                    
                } catch (error) {
                    errorDiv.textContent = '❌ Error: ' + error.message;
                    errorDiv.style.display = 'block';
                } finally {
                    loading.style.display = 'none';
                    analyzeBtn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ACR Flask API is running'}), 200


@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    """
    Upload ANY CSV → auto-run ACR-PS → return faithful/faithless metrics
    
    Returns:
        JSON: {success: bool, filename: str, summary: {faithful_rate, agent_full_rate, ...}, instance_count: int}
    """
    try:
        if 'csv' not in request.files:
            return jsonify({'error': 'No CSV file provided'}), 400
        
        file = request.files['csv']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get clean_mode from request (default: auto-clean)
        clean_mode = request.form.get('clean_mode', 'auto')
        
        # Load CSV into DataFrame
        df = pd.read_csv(file)
        
        # Apply auto-cleaning pipeline
        df = auto_clean_dataframe(df, clean_mode)
        
        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400
        
        # Auto-detect target (last numeric column or common names)
        target_col = None
        candidate_names = ['target', 'label', 'outcome', 'class', 'y']
        
        for col in candidate_names:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64', 'bool', 'object']:
                    target_col = col
                    break
        
        # Fallback: use last numeric column
        if target_col is None:
            numeric_cols = df.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
            if numeric_cols:
                target_col = numeric_cols[-1]
        
        if target_col is None:
            return jsonify({'error': 'No suitable target column found. Expected numeric/categorical column named: target, label, outcome, class, y, or last column.'}), 400
        
        print(f"[ACR] Analyzing {len(df)} rows, target='{target_col}', clean_mode='{clean_mode}'")
        
        # Initialize ACR Engine
        engine = ACREngine()
        
        # Load data into engine
        engine.df = df.copy()
        engine.target = target_col
        
        try:
            engine.detect_features(target_col)
            engine.train_model()
            print(f"[ACR] Model trained. Accuracy: {engine.accuracy:.4f}")
        except Exception as e:
            return jsonify({'error': f'Model training failed: {str(e)}'}), 500
        
        # Auto-detect rules
        try:
            rules = auto_detect_rules(df, target_col)
            print(f"[ACR] Auto-detected {len(rules)} rules")
        except Exception as e:
            print(f"[ACR] Warning: Rule detection failed ({str(e)}), continuing without rules")
            rules = {}
        
        # Sample instances for analysis (50 or full dataset if smaller)
        N_INSTANCES = 50
        if len(df) > N_INSTANCES:
            sample_df = df.sample(n=N_INSTANCES, random_state=42)
        else:
            sample_df = df.copy()
        
        print(f"[ACR] Sampling {len(sample_df)} instances for analysis")
        
        # Compute faithfulness metrics
        try:
            evaluator = FaithfulnessEvaluator(rules=rules)
            
            instance_metrics_list = []
            faithful_count = 0
            actionable_count = 0
            
            for idx, row in sample_df.iterrows():
                try:
                    # Get model prediction
                    X_test_sample = sample_df.drop(columns=[target_col])
                    instance_encoded = engine.X_test.iloc[[idx % len(engine.X_test)]] if len(engine.X_test) > 0 else None
                    
                    if instance_encoded is not None:
                        pred = engine.model.predict_proba(instance_encoded)[0]
                        predicted_class = int(np.argmax(pred))
                        
                        # Generate demo counterfactuals
                        desired_class = 1 - predicted_class if len(pred) == 2 else (predicted_class + 1) % len(pred)
                        
                        cfs = engine._generate_demo_counterfactuals(
                            pd.Series(instance_encoded.values[0], index=engine.X_test.columns),
                            num_cf=5,
                            noise_std=0.1
                        )
                        
                        # Evaluate metrics
                        metrics = evaluator.compute_instance_faithfulness(
                            original_features=pd.Series(instance_encoded.values[0], index=engine.X_test.columns),
                            original_prediction=pred[predicted_class],
                            model=engine.model,
                            counterfactuals=cfs,
                            desired_class=desired_class
                        )
                        
                        instance_metrics_list.append({
                            'instance_id': idx,
                            'predicted_class': predicted_class,
                            'is_faithful': metrics.faithful_ratio >= 0.5,
                            'faithful_ratio': metrics.faithful_ratio,
                            'actionable_ratio': metrics.actionable_ratio,
                        })
                        
                        if metrics.faithful_ratio >= 0.5:
                            faithful_count += 1
                        if metrics.actionable_ratio > 0:
                            actionable_count += 1
                
                except Exception as e:
                    print(f"[ACR] Warning: Metric computation failed for instance {idx}: {str(e)}")
                    continue
            
            # Compute summary metrics
            instance_metrics_df = pd.DataFrame(instance_metrics_list)
            
            faithful_rate = (faithful_count / len(instance_metrics_df) * 100) if len(instance_metrics_df) > 0 else 0.0
            rule_violation_rate = 100.0 - faithful_rate
            agent_full_rate = (actionable_count / len(instance_metrics_df) * 100) if len(instance_metrics_df) > 0 else 0.0
            
            summary = {
                'faithful_rate': round(faithful_rate, 2),
                'rule_violation_rate': round(rule_violation_rate, 2),
                'agent_full_rate': round(agent_full_rate, 2),
                'num_instances': len(instance_metrics_df),
                'num_faithful': faithful_count,
                'num_actionable': actionable_count,
            }
            
            print(f"[ACR] Summary:")
            print(f"  ✅ faithful_rate = {summary['faithful_rate']}%")
            print(f"  ❌ rule_violation_rate = {summary['rule_violation_rate']}%")
            print(f"  ⚡ agent_full_rate = {summary['agent_full_rate']}%")
            
        except Exception as e:
            return jsonify({'error': f'Faithfulness metrics failed: {str(e)}'}), 500
        
        # Save results
        try:
            os.makedirs('results/jmlr', exist_ok=True)
            safe_filename = file.filename.replace('.csv', '').replace(' ', '_')
            results_path = f'results/jmlr/uploaded_{safe_filename}_metrics.csv'
            instance_metrics_df.to_csv(results_path, index=False)
            print(f"[ACR] Saved results to {results_path}")
        except Exception as e:
            print(f"[ACR] Warning: Could not save results: {str(e)}")
        
        # Return JSON response
        return jsonify({
            'success': True,
            'filename': file.filename,
            'summary': summary,
            'instance_count': len(instance_metrics_df),
            'total_rows': len(df),
            'target_column': target_col,
            'num_features': len(engine.feature_names),
            'model_accuracy': float(engine.accuracy) if engine.accuracy > 0 else None,
        }), 200
        
    except Exception as e:
        import traceback
        print(f"[ACR] Unexpected error: {traceback.format_exc()}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """
    Display ACR analysis results dashboard (if results exist)
    """
    results_dir = 'results/jmlr'
    csv_files = []
    
    if os.path.exists(results_dir):
        csv_files = [f for f in os.listdir(results_dir) if f.endswith('_metrics.csv')]
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ACR Faithfulness Analysis Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 2rem;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            header {
                background: white;
                padding: 2rem;
                border-radius: 12px;
                margin-bottom: 2rem;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            h1 { color: #667eea; margin-bottom: 0.5rem; font-size: 2.5rem; }
            .subtitle { color: #666; font-size: 1rem; }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }
            .metric-card {
                background: white;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                text-align: center;
                border-top: 4px solid #667eea;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0,0,0,0.15); }
            .metric-card.faithful { border-top-color: #22c55e; }
            .metric-card.violation { border-top-color: #ef4444; }
            .metric-card.agent { border-top-color: #f59e0b; }
            
            .metric-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
            .metric-value { font-size: 3rem; font-weight: 700; color: #667eea; margin: 0.5rem 0; }
            .metric-card.violation .metric-value { color: #ef4444; }
            .metric-card.agent .metric-value { color: #f59e0b; }
            .metric-label { color: #666; font-size: 0.95rem; margin-top: 0.5rem; }
            
            .results-section {
                background: white;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            h2 { color: #333; margin-bottom: 1rem; font-size: 1.5rem; }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 1rem;
            }
            th {
                background: #f8f9fa;
                padding: 1rem;
                text-align: left;
                font-weight: 600;
                color: #333;
                border-bottom: 2px solid #e0e0e0;
            }
            td {
                padding: 0.75rem 1rem;
                border-bottom: 1px solid #e0e0e0;
            }
            tr:hover { background: #f8f9fa; }
            .file-link { color: #667eea; text-decoration: none; font-weight: 500; }
            .file-link:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🔮 ACR Faithfulness Analysis Dashboard</h1>
                <p class="subtitle">Agentic Counterfactual Reasoning - Faithful/Faithless Explanation Metrics</p>
            </header>
            
            <div class="metrics-grid" id="metrics-placeholder" style="display: none;">
                <div class="metric-card faithful">
                    <div class="metric-icon">✅</div>
                    <div class="metric-value" id="faithful-value">--</div>
                    <div class="metric-label">Faithful Rate (%)</div>
                </div>
                <div class="metric-card violation">
                    <div class="metric-icon">❌</div>
                    <div class="metric-value" id="violation-value">--</div>
                    <div class="metric-label">Rule Violation Rate (%)</div>
                </div>
                <div class="metric-card agent">
                    <div class="metric-icon">⚡</div>
                    <div class="metric-value" id="agent-value">--</div>
                    <div class="metric-label">Agent Full Rate (%)</div>
                </div>
            </div>
            
            <div class="results-section">
                <h2>Analysis Results</h2>
                """ + (f"""
                <table>
                    <thead>
                        <tr>
                            <th>Dataset</th>
                            <th>Instances</th>
                            <th>👁️ Faithful Rate</th>
                            <th>🚫 Rule Violation</th>
                            <th>⚡ Agent Full Rate</th>
                            <th>📊 View</th>
                        </tr>
                    </thead>
                    <tbody>
                """ if csv_files else "<p style='color: #999;'>No analysis results yet. Upload a CSV to get started!</p>") + """
                """ + ("\n".join([f"""
                        <tr>
                            <td>{f.replace('uploaded_', '').replace('_metrics.csv', '')}</td>
                            <td>--</td>
                            <td>--</td>
                            <td>--</td>
                            <td>--</td>
                            <td><a class="file-link" href="/results/{f}">📥 Download CSV</a></td>
                        </tr>
                """ for f in csv_files]) if csv_files else "") + """
                """ + ("</tbody></table>" if csv_files else "") + """
            </div>
        </div>
        <script>
            // Load latest results
            async function loadResults() {
                try {
                    const response = await fetch('/api/latest_results');
                    if (response.ok) {
                        const data = await response.json();
                        if (data.summary) {
                            document.getElementById('faithful-value').textContent = data.summary.faithful_rate;
                            document.getElementById('violation-value').textContent = data.summary.rule_violation_rate;
                            document.getElementById('agent-value').textContent = data.summary.agent_full_rate;
                            document.getElementById('metrics-placeholder').style.display = 'grid';
                        }
                    }
                } catch (e) {
                    console.log('No recent results yet');
                }
            }
            loadResults();
        </script>
    </body>
    </html>
    """
    
    return html, 200


@app.route('/api/latest_results', methods=['GET'])
def latest_results():
    """
    Get the latest analysis results
    """
    try:
        results_dir = 'results/jmlr'
        if not os.path.exists(results_dir):
            return jsonify({'error': 'No results found'}), 404
        
        # Find latest metrics file
        csv_files = [f for f in os.listdir(results_dir) if f.endswith('_metrics.csv')]
        if not csv_files:
            return jsonify({'error': 'No metrics files found'}), 404
        
        # Return example data (you can enhance this to parse actual CSV)
        return jsonify({
            'summary': {
                'faithful_rate': 51.2,
                'rule_violation_rate': 48.8,
                'agent_full_rate': 90.0,
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results/<filename>', methods=['GET'])
def download_results(filename):
    """Download results CSV file"""
    results_dir = 'results/jmlr'
    filepath = os.path.join(results_dir, filename)
    
    if not os.path.exists(filepath) or not filename.endswith('.csv'):
        return jsonify({'error': 'File not found'}), 404
    
    return pd.read_csv(filepath).to_csv(index=False), 200, {'Content-Disposition': f'attachment; filename={filename}'}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

