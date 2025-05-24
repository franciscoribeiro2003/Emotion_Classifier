from emotion_classifier.config import PROJECT_ROOT as project_root


import pickle
import json
from pathlib import Path

# Simple result persistence functions
def save_results(model_name, report, training_time, labels_and_preds):
    """Save results for a single model"""
    results_dir = project_root / "experiment_results"
    results_dir.mkdir(exist_ok=True)
    
    # Save classification report
    with open(results_dir / f"{model_name}_report.pkl", 'wb') as f:
        pickle.dump(report, f)
    
    # Save training time
    with open(results_dir / f"{model_name}_time.json", 'w') as f:
        json.dump({"training_time": training_time}, f)
    
    # Save predictions and labels
    with open(results_dir / f"{model_name}_predictions.pkl", 'wb') as f:
        pickle.dump(labels_and_preds, f)
    
    print(f"✅ Results saved for {model_name}")

def load_all_results():
    """Load all saved results"""
    results_dir = project_root / "experiment_results"
    
    model_results = {}
    training_times = {}
    detailed_reports = {}
    
    if not results_dir.exists():
        print("No saved results found")
        return model_results, training_times, detailed_reports
    
    # Find all saved models
    for report_file in results_dir.glob("*_report.pkl"):
        model_name = report_file.stem.replace("_report", "")
        
        try:
            # Load report
            with open(report_file, 'rb') as f:
                model_results[model_name] = pickle.load(f)
            
            # Load training time
            time_file = results_dir / f"{model_name}_time.json"
            if time_file.exists():
                with open(time_file, 'r') as f:
                    training_times[model_name] = json.load(f)["training_time"]
            
            # Load predictions
            pred_file = results_dir / f"{model_name}_predictions.pkl"
            if pred_file.exists():
                with open(pred_file, 'rb') as f:
                    detailed_reports[model_name] = pickle.load(f)
            
            print(f"✅ Loaded results for {model_name}")
            
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
    
    return model_results, training_times, detailed_reports

# Initialize storage dictionaries
model_results = {}
training_times = {}
detailed_reports = {}

print("Simple persistence system ready!")